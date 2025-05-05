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

import itertools
from functools import partial, wraps

import numpy as np
import torch

from opencomplex.np import nucleotide_constants as nc
from opencomplex.np import residue_constants as rc
from opencomplex.utils.complex_utils import (
    ComplexType,
    correct_rna_butype,
    split_protein_rna_pos,
)
from opencomplex.utils.rigid_utils import Rigid, Rotation
from opencomplex.utils.tensor_utils import (
    batched_gather,
    map_padcat,
    padcat,
    padto,
)


def curry1(f):
    """Supply all arguments but the first."""

    @wraps(f)
    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def cast_to_64bit_ints(bio_complex):
    # We keep all ints as int64
    for k, v in bio_complex.items():
        if v.dtype == torch.int32:
            bio_complex[k] = v.type(torch.int64)

    return bio_complex


def cast_to_32bit_floats(bio_complex):
    # We keep all floats at most float32
    for k, v in bio_complex.items():
        if v.dtype == torch.float64:
            bio_complex[k] = v.type(torch.float32)

    return bio_complex


def make_one_hot(x, num_classes):
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1).type(torch.int64), 1)
    return x_one_hot


@curry1
def split_pos(bio_complex, complex_type):
    bio_complex["protein_pos"], bio_complex["rna_pos"] = split_protein_rna_pos(
        bio_complex, complex_type
    )
    return bio_complex


def make_seq_mask(bio_complex):
    bio_complex["seq_mask"] = torch.ones(
        bio_complex["butype"].shape, dtype=torch.float32
    )

    # unknown type X
    bio_complex["seq_mask"][
        torch.where(bio_complex["butype"] == bio_complex["num_butype"])
    ] = 0.0

    return bio_complex


def squeeze_features(bio_complex):
    """Remove singleton and repeated dimensions in bio_complex features."""
    if bio_complex["butype"].ndim == 2:
        # NOTE: data format capability
        bio_complex["butype"] = torch.argmax(bio_complex["butype"], dim=-1)
    if "template_butype" in bio_complex and bio_complex["template_butype"].ndim > 2:
        bio_complex["template_butype"] = torch.argmax(
            bio_complex["template_butype"], dim=-1
        )
    for k in [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        # "resolution",
        "between_segment_residues",
        "residue_index",
        "template_all_atom_mask",
    ]:
        if k in bio_complex:
            final_dim = bio_complex[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(bio_complex[k]):
                    bio_complex[k] = torch.squeeze(bio_complex[k], dim=-1)
                else:
                    bio_complex[k] = np.squeeze(bio_complex[k], axis=-1)

    for k in ["seq_length", "num_alignments"]:
        if k in bio_complex:
            bio_complex[k] = bio_complex[k][0]

    return bio_complex


def pseudo_beta_fn(
    bio_complex, butype, all_atom_positions, all_atom_mask, complex_type=None
):
    """Create pseudo beta features."""
    if "protein_pos" in bio_complex:
        protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]
    else:
        protein_pos, rna_pos = split_protein_rna_pos(bio_complex, complex_type)
    is_gly = torch.eq(butype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    o4_idx = nc.atom_order["O4'"]
    pseudo_beta = padcat(
        [
            torch.where(
                torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
                all_atom_positions[..., ca_idx, :],
                all_atom_positions[..., cb_idx, :],
            )[..., protein_pos, :],
            all_atom_positions[..., rna_pos, o4_idx, :],
        ],
        axis=-2,
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = padcat(
            [
                torch.where(
                    is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
                )[..., protein_pos],
                all_atom_mask[..., rna_pos, o4_idx],
            ],
            axis=-1,
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


@curry1
def make_pseudo_beta(bio_complex, prefix=""):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ["", "template_"]
    (
        bio_complex[prefix + "pseudo_beta"],
        bio_complex[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        bio_complex,
        bio_complex[prefix + "butype"],
        bio_complex[prefix + "all_atom_positions"],
        bio_complex[prefix + "all_atom_mask"],
    )
    return bio_complex


@curry1
def make_dense_atom_masks(bio_complex, complex_type):
    device = bio_complex["butype"].device

    protein_map = rc.get_restype_atom_mapping(device)
    rna_map = nc.get_restype_atom_mapping(device)
    if complex_type == ComplexType.PROTEIN:
        mapping = protein_map
    elif complex_type == ComplexType.RNA:
        mapping = rna_map
    else:
        mapping = map_padcat(protein_map, rna_map)

    # Add mapping for unknown type
    padding = partial(torch.nn.functional.pad, pad=(0, 0, 0, 1))
    (
        restype_dense_to_all,
        restype_all_to_dense,
        restype_dense_mask,
        restype_all_mask,
    ) = map(padding, mapping)

    butype = bio_complex["butype"].to(torch.long)

    # create the mapping for (residx, atom_dense) --> all, i.e. an array
    # with shape (num_res, dense) containing the all indices for this bio_complex
    residx_dense_to_all = restype_dense_to_all[butype]
    residx_dense_mask = restype_dense_mask[butype]

    bio_complex["dense_atom_exists"] = residx_dense_mask
    bio_complex["residx_dense_to_all"] = residx_dense_to_all.long()

    # create the gather indices for mapping back
    residx_all_to_dense = restype_all_to_dense[butype]
    bio_complex["residx_all_to_dense"] = residx_all_to_dense.long()

    residx_all_mask = restype_all_mask[butype]
    bio_complex["all_atom_exists"] = residx_all_mask

    return bio_complex


def make_dense_atom_positions(bio_complex):
    residx_dense_mask = bio_complex["dense_atom_exists"]
    residx_dense_to_all = bio_complex["residx_dense_to_all"]

    # Create a mask for known ground truth positions.
    residx_dense_gt_mask = residx_dense_mask * batched_gather(
        bio_complex["all_atom_mask"],
        residx_dense_to_all,
        dim=-1,
        no_batch_dims=len(bio_complex["all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_dense_gt_positions = residx_dense_gt_mask[..., None] * (
        batched_gather(
            bio_complex["all_atom_positions"],
            residx_dense_to_all,
            dim=-2,
            no_batch_dims=len(bio_complex["all_atom_positions"].shape[:-2]),
        )
    )

    bio_complex["dense_atom_exists"] = residx_dense_mask
    bio_complex["dense_atom_gt_exists"] = residx_dense_gt_mask
    bio_complex["dense_atom_gt_positions"] = residx_dense_gt_positions

    bio_complex = compute_alt_gt(bio_complex)

    return bio_complex


def compute_alt_gt(bio_complex):
    protein_pos = bio_complex["protein_pos"]
    butype = torch.clamp(bio_complex["butype"][protein_pos], max=20)

    restype_3 = [rc.restype_1to3[res] for res in rc.restypes]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype=bio_complex["all_atom_mask"].dtype,
            device=bio_complex["all_atom_mask"].device,
        )
        for res in restype_3
    }
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(14, device=bio_complex["all_atom_mask"].device)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = rc.restype_name_to_atom14_names[resname].index(
                source_atom_swap
            )
            target_index = rc.restype_name_to_atom14_names[resname].index(
                target_atom_swap
            )
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = bio_complex["all_atom_mask"].new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix

    renaming_matrices = torch.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[butype]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum(
        "...rac,...rab->...rbc",
        bio_complex["dense_atom_gt_positions"][protein_pos, :14],
        renaming_transform,
    )
    bio_complex["dense_atom_alt_gt_positions"] = padto(
        alternative_gt_positions, bio_complex["dense_atom_gt_positions"].shape
    )

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum(
        "...ra,...rab->...rb",
        bio_complex["dense_atom_gt_exists"][protein_pos, :14],
        renaming_transform,
    )
    bio_complex["dense_atom_alt_gt_exists"] = padto(
        alternative_gt_mask, bio_complex["dense_atom_gt_exists"].shape
    )

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = bio_complex["all_atom_mask"].new_zeros((21, 14))
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = rc.restype_order[rc.restype_3to1[resname]]
            atom_idx1 = rc.restype_name_to_atom14_names[resname].index(atom_name1)
            atom_idx2 = rc.restype_name_to_atom14_names[resname].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    bio_complex["dense_atom_is_ambiguous"] = padto(
        restype_atom14_is_ambiguous[butype], bio_complex["dense_atom_gt_exists"].shape
    )
    return bio_complex


def atom37_to_frames(butype, all_atom_positions, all_atom_mask, eps=1e-8):
    batch_dims = len(butype.shape[:-1])
    butype = torch.clamp(butype, max=20)

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[restype, chi_idx + 4, :] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*butype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(rc.chi_angles_mask)

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = butype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = restype_rigidgroup_base_atom37_idx.view(
        *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        butype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    base_atom_pos = batched_gather(
        all_atom_positions,
        residx_rigidgroup_base_atom37_idx,
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )

    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
    )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        butype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=butype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 21, 8
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=butype.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        butype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        butype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = Rotation(rot_mats=residx_rigidgroup_ambiguity_rot)
    alt_gt_frames = gt_frames.compose(Rigid(residx_rigidgroup_ambiguity_rot, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    return (
        gt_frames_tensor,
        gt_exists,
        group_exists,
        residx_rigidgroup_is_ambiguous,
        alt_gt_frames_tensor,
    )


def atom27_to_frames(butype, all_atom_positions, all_atom_mask, eps=1e-8):
    butype = correct_rna_butype(butype)

    batch_dims = len(butype.shape[:-1])
    # 5 NT types (AUGC plus X), 9 groups
    nttype_rigidgroup_base_atom_names = np.full([5, 9, 3], "", dtype=object)
    # atoms that constitute backbone frame 1
    nttype_rigidgroup_base_atom_names[:, 0, :] = ["O4'", "C4'", "C3'"]
    nttype_rigidgroup_base_atom_names[:, 1, :] = ["O4'", "C1'", "C2'"]

    for restype, restype_letter in enumerate(nc.restypes):
        # keep one-letter format in RNA
        resname = restype_letter
        for torsion_idx in range(7):
            if nc.chi_angles_mask[restype][torsion_idx]:
                names = nc.chi_angles_atoms[resname][torsion_idx]
                nttype_rigidgroup_base_atom_names[restype, torsion_idx + 2, :] = names[
                    1:
                ]

    # Or can be initiazed in all_ones for previous 4 dims as there are no missing frames in RNA
    nttype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*butype.shape[:-1], 5, 9),
    )
    nttype_rigidgroup_mask[..., 0] = 1
    nttype_rigidgroup_mask[..., 1] = 1
    nttype_rigidgroup_mask[..., :4, 2:] = all_atom_mask.new_tensor(nc.chi_angles_mask)

    lookuptable = nc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    # get atom index in atom_types that defined in nucleotide_constants
    nttype_rigidgroup_base_atom27_idx = lookup(
        nttype_rigidgroup_base_atom_names,
    )
    # 5 (nt types) * 7 (torsions) * 3 (frame atom indexs)
    nttype_rigidgroup_base_atom27_idx = butype.new_tensor(
        nttype_rigidgroup_base_atom27_idx,
    )
    # # 1 * 5 (nt types) * 7 (torsions) * 3 (frame atom indexs)
    nttype_rigidgroup_base_atom27_idx = nttype_rigidgroup_base_atom27_idx.view(
        *((1,) * batch_dims), *nttype_rigidgroup_base_atom27_idx.shape
    )
    # # N * 5 (nt types) * 7 (torsions) * 3 (frame atom indexs)
    ntidx_rigidgroup_base_atom27_idx = batched_gather(
        nttype_rigidgroup_base_atom27_idx,
        butype.to(torch.long),
        dim=-3,
        no_batch_dims=batch_dims,
    )
    base_atom_pos = batched_gather(
        all_atom_positions,
        ntidx_rigidgroup_base_atom27_idx.to(torch.long),
        dim=-2,
        no_batch_dims=len(all_atom_positions.shape[:-2]),
    )
    # # 0, 1, 2 are the index of frame atoms
    gt_frames = Rigid.from_3_points(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=1e-8,
    )

    group_exists = batched_gather(
        nttype_rigidgroup_mask,
        butype.type(torch.long),
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        ntidx_rigidgroup_base_atom27_idx.to(torch.long),
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=butype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 9, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1
    rots = Rotation(rot_mats=rots)

    gt_frames = gt_frames.compose(Rigid(rots, None))

    gt_frames_tensor = gt_frames.to_tensor_4x4()

    residx_rigidgroup_is_ambiguous = torch.zeros_like(group_exists)
    alt_gt_frames_tensor = torch.zeros_like(gt_frames_tensor)

    return (
        gt_frames_tensor,
        gt_exists,
        group_exists,
        residx_rigidgroup_is_ambiguous,
        alt_gt_frames_tensor,
    )


@curry1
def all_atom_to_frames(bio_complex, eps=1e-8):
    butype = bio_complex["butype"]
    all_atom_positions = bio_complex["all_atom_positions"]
    all_atom_mask = bio_complex["all_atom_mask"]
    protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]

    (
        bio_complex["rigidgroups_gt_frames"],
        bio_complex["rigidgroups_gt_exists"],
        bio_complex["rigidgroups_group_exists"],
        bio_complex["rigidgroups_group_is_ambiguous"],
        bio_complex["rigidgroups_alt_gt_frames"],
    ) = map_padcat(
        atom37_to_frames(
            butype[protein_pos],
            all_atom_positions[protein_pos],
            all_atom_mask[protein_pos],
            eps,
        ),
        atom27_to_frames(
            butype[rna_pos], all_atom_positions[rna_pos], all_atom_mask[rna_pos], eps
        ),
    )

    return bio_complex


def get_chi_atom_indices(complex_type):
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.butypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    if complex_type == "protein":
        for residue_name in rc.restypes:
            residue_name = rc.restype_1to3[residue_name]
            residue_chi_angles = rc.chi_angles_atoms[residue_name]
            atom_indices = []
            for chi_angle in residue_chi_angles:
                atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
            for _ in range(4 - len(atom_indices)):
                atom_indices.append(
                    [0, 0, 0, 0]
                )  # For chi angles not defined on the AA.
            chi_atom_indices.append(atom_indices)

        chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.
    else:
        for residue_name in nc.restypes:
            residue_chi_angles = nc.chi_angles_atoms[residue_name]
            atom_indices = []
            for chi_angle in residue_chi_angles:
                atom_indices.append([nc.atom_order[atom] for atom in chi_angle])
            for _ in range(7 - len(atom_indices)):
                atom_indices.append(
                    [0, 0, 0, 0]
                )  # For chi angles not defined on the NT.
            chi_atom_indices.append(atom_indices)

        chi_atom_indices.append([[0, 0, 0, 0]] * 7)  # For UNKNOWN residue.

    return chi_atom_indices


def atom37_to_torsion_angles(
    butype,
    all_atom_positions,
    all_atom_mask,
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)butype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    if butype.shape[-1] == 0:
        return None, None, None

    butype = torch.clamp(butype, max=20)

    num_atoms = all_atom_mask.shape[-1]

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, num_atoms, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, num_atoms])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(prev_all_atom_mask[..., 1:3], dim=-1) * torch.prod(
        all_atom_mask[..., :2], dim=-1
    )
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices("protein"), device=butype.device
    )

    atom_indices = chi_atom_indices[..., butype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[butype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[butype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*butype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )

    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask


def atom27_to_torsion_angles(
    butype,
    all_atom_positions,
    all_atom_mask,
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 27, 3] atom positions (in atom27
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 27] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    butype = correct_rna_butype(butype)
    butype = torch.clamp(butype, max=4)

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices("RNA"), device=butype.device
    )

    atom_indices = chi_atom_indices[..., butype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(nc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[butype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask
    # In the order of delta. gamma, beta, alpha1, alpha2, tm, chi
    torsions_atom_pos = chis_atom_pos

    torsion_angles_mask = chis_mask

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = (
        torsion_angles_sin_cos
        * all_atom_mask.new_tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    )
    alt_torsion_angles_sin_cos = torch.zeros_like(torsion_angles_sin_cos)

    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask


@curry1
def all_atom_to_torsion_angles(
    bio_complex,
    prefix="",
):
    butype = bio_complex[prefix + "butype"]
    all_atom_positions = bio_complex[prefix + "all_atom_positions"]
    all_atom_mask = bio_complex[prefix + "all_atom_mask"]

    protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]
    (
        bio_complex[prefix + "torsion_angles_sin_cos"],
        bio_complex[prefix + "alt_torsion_angles_sin_cos"],
        bio_complex[prefix + "torsion_angles_mask"],
    ) = map_padcat(
        atom37_to_torsion_angles(
            butype[..., protein_pos],
            all_atom_positions[..., protein_pos, :, :],
            all_atom_mask[..., protein_pos, :],
        ),
        atom27_to_torsion_angles(
            butype[..., rna_pos],
            all_atom_positions[..., rna_pos, :27, :],
            all_atom_mask[..., rna_pos, :27],
        ),
        axis=butype.ndim - 1,
    )

    return bio_complex


def get_backbone_frames(bio_complex):
    protein_pos, rna_pos = bio_complex["protein_pos"], bio_complex["rna_pos"]
    bio_complex["backbone_rigid_tensor"] = padcat(
        [
            bio_complex["rigidgroups_gt_frames"][protein_pos, 0:1, :, :],
            bio_complex["rigidgroups_gt_frames"][rna_pos, 0:2, :, :],
        ],
    )
    bio_complex["backbone_rigid_mask"] = padcat(
        [
            bio_complex["rigidgroups_gt_exists"][protein_pos, 0:1],
            bio_complex["rigidgroups_gt_exists"][rna_pos, 0:2],
        ]
    )

    return bio_complex


def get_chi_angles(bio_complex):
    dtype = bio_complex["all_atom_mask"].dtype
    bio_complex["chi_angles_sin_cos"] = bio_complex["torsion_angles_sin_cos"].to(dtype)
    bio_complex["chi_mask"] = bio_complex["torsion_angles_mask"].to(dtype)
    return bio_complex

