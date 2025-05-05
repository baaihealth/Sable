# Copyright 2025 Beijing Academy of Artificial Intelligence (BAAI)
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import ml_collections
import torch

from einops import repeat

from opencomplex.loss.loss_utils import lddt

from opencomplex.utils.rigid_utils import Rigid


def compute_renamed_ground_truth(
    dense_pred_positions: torch.Tensor,
    dense_atom_gt_positions: torch.Tensor,
    dense_atom_gt_exists: torch.Tensor,
    dense_atom_alt_gt_positions: torch.Tensor = None,
    dense_atom_is_ambiguous: torch.Tensor = None,
    dense_atom_alt_gt_exists: torch.Tensor = None,
    eps=1e-10,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      atom14_pred_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """
    if dense_atom_alt_gt_positions is None:
        return {
            "alt_naming_is_better": None,
            "renamed_dense_atom_gt_positions": dense_atom_gt_positions,
            "renamed_dense_atom_gt_exists": dense_atom_gt_exists,
        }

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                dense_pred_positions[..., None, :, None, :]
                - dense_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                dense_atom_gt_positions[..., None, :, None, :]
                - dense_atom_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                dense_atom_alt_gt_positions[..., None, :, None, :]
                - dense_atom_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    mask = (
        dense_atom_gt_exists[..., None, :, None]
        * dense_atom_is_ambiguous[..., None, :, None]
        * dense_atom_gt_exists[..., None, :, None, :]
        * (1.0 - dense_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))

    fp_type = dense_pred_positions.dtype
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)

    renamed_atom14_gt_positions = (
        1.0 - alt_naming_is_better[..., None, None]
    ) * dense_atom_gt_positions + alt_naming_is_better[
        ..., None, None
    ] * dense_atom_alt_gt_positions

    renamed_atom14_gt_mask = (
        1.0 - alt_naming_is_better[..., None]
    ) * dense_atom_gt_exists + alt_naming_is_better[
        ..., None
    ] * dense_atom_alt_gt_exists

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_dense_atom_gt_positions": renamed_atom14_gt_positions,
        "renamed_dense_atom_gt_exists": renamed_atom14_gt_mask,
    }


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    pair_mask: torch.Tensor = None,
    eps=1e-8,
) -> torch.Tensor:
    """
    Computes FAPE loss.

    Args:
        pred_frames:
            [*, N_frames] Rigid object of predicted frames
        target_frames:
            [*, N_frames] Rigid object of ground truth frames
        frames_mask:
            [*, N_frames] binary mask for the frames
        pred_positions:
            [*, N_pts, 3] predicted atom positions
        target_positions:
            [*, N_pts, 3] ground truth positions
        positions_mask:
            [*, N_pts] positions mask
        length_scale:
            Length scale by which the loss is divided
        l1_clamp_distance:
            Cutoff above which distance errors are disregarded
        pair_mask:
            A (num_frames, num_positions) mask to use in the loss, useful
            for separating intra from inter chain losses.
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]
    if pair_mask is not None:
        normed_error *= pair_mask[..., None, :, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)

    if pair_mask is not None:
        normed_error = torch.sum(normed_error, dim=(-2, -1))
        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask[..., None, :, :]
        normalization_factor = torch.sum(mask, dim=(-2, -1))
        normed_error = normed_error / (eps + normalization_factor)
    else:
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_rigid_tensor: torch.Tensor,
    backbone_rigid_mask: torch.Tensor,
    pred_aff: Rigid,
    enable_use_clamped_fape: bool = False,
    use_clamped_fape: Optional[torch.Tensor] = None,
    pair_mask: torch.Tensor = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    gt_aff = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    if backbone_rigid_mask.shape[-1] > 1:
        # concat in N dim, from [*, N, 2] to [*, N*2]
        gt_aff = Rigid.cat([gt_aff[..., 0], gt_aff[..., 1]], dim=-1)
        pred_aff = Rigid.cat([pred_aff[..., 0], pred_aff[..., 1]], dim=-1)
        backbone_rigid_mask = torch.cat(
            [backbone_rigid_mask[..., 0], backbone_rigid_mask[..., 1]], dim=-1
        )

        if pair_mask is not None:
            pair_mask = repeat(pair_mask, "... h w -> ... (a h) (b w)", a=2, b=2)
    else:
        backbone_rigid_mask = backbone_rigid_mask[..., 0]
        gt_aff = gt_aff[..., 0]

    fape_loss = compute_fape(
        pred_aff,
        # NOTE(yujingcheng): wrong squeeze dim in batch training
        gt_aff[:, None, :],
        backbone_rigid_mask[:, None, :],
        pred_aff.get_trans(),
        gt_aff[:, None, :].get_trans(),
        backbone_rigid_mask[:, None, :],
        pair_mask=pair_mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    if enable_use_clamped_fape and use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_rigid_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_rigid_mask[None],
            pair_mask=pair_mask,
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss


def sidechain_loss(
    sidechain_frames: torch.Tensor,  # predicted frames
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,  # ground truth frames
    rigidgroups_gt_exists: torch.Tensor,
    renamed_dense_atom_gt_positions: torch.Tensor,
    renamed_dense_atom_gt_exists: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor = None,
    alt_naming_is_better: torch.Tensor = None,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    sidechain_frames = sidechain_frames[-1]

    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)

    if alt_naming_is_better is not None:
        renamed_gt_frames = (
            1.0 - alt_naming_is_better[..., None, None, None]
        ) * rigidgroups_gt_frames + alt_naming_is_better[
            ..., None, None, None
        ] * rigidgroups_alt_gt_frames
    else:
        renamed_gt_frames = rigidgroups_gt_frames

    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)

    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)

    sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_dense_atom_gt_positions = renamed_dense_atom_gt_positions.view(
        *batch_dims, -1, 3
    )
    renamed_dense_atom_gt_exists = renamed_dense_atom_gt_exists.view(*batch_dims, -1)

    fape = compute_fape(
        pred_frames=sidechain_frames,
        target_frames=renamed_gt_frames,
        frames_mask=rigidgroups_gt_exists,
        pred_positions=sidechain_atom_pos,
        target_positions=renamed_dense_atom_gt_positions,
        positions_mask=renamed_dense_atom_gt_exists,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape


def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
    asym_id: torch.Tensor = None,
    disable_sc: bool = False,
) -> torch.Tensor:
    break_down = {}
    if asym_id is not None:
        intra_chain_mask = (asym_id[..., :, None] == asym_id[..., None, :]).float()
        intra_chain_bb_loss = backbone_loss(
            pred_aff=out["sm"]["frames"],
            pair_mask=intra_chain_mask,
            **{**batch, **config["intra_chain_backbone"]},
        )
        interface_bb_loss = backbone_loss(
            pred_aff=out["sm"]["frames"],
            pair_mask=1.0 - intra_chain_mask,
            **{**batch, **config["interface_backbone"]},
        )
        bb_loss = (
            config["intra_chain_backbone"]["weight"] * intra_chain_bb_loss
            + config["interface_backbone"]["weight"] * interface_bb_loss
        )
        break_down = {
            "bb_fape": torch.mean(bb_loss),
            "intra_chain_bb_fape": torch.mean(intra_chain_bb_loss),
            "interface_bb_fape": torch.mean(interface_bb_loss),
        }
    else:
        bb_loss = backbone_loss(
            pred_aff=out["sm"]["frames"],
            **{**batch, **config["backbone"]},
        )
        break_down = {
            "bb_fape": torch.mean(bb_loss),
            "intra_chain_bb_fape": torch.mean(bb_loss),
            "interface_bb_fape": torch.mean(torch.zeros_like(bb_loss)),
        }

    if disable_sc:
        sc_loss = out["sm"]["sidechain_frames"].sum() * 0.0
    else:
        sc_loss = sidechain_loss(
            out["sm"]["sidechain_frames"],
            out["sm"]["positions"],
            **compute_renamed_ground_truth(
                dense_pred_positions=out["sm"]["positions"][-1], **batch
            ),
            **{**batch, **config["sidechain"]},
        )
    break_down["sidechain_fape"] = torch.mean(sc_loss)

    loss = config["backbone"]["weight"] * bb_loss + config["sidechain"]["weight"] * sc_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss, break_down

