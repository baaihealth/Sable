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

import torch

from opencomplex.utils.tensor_utils import permute_final_dims


def dist_map_to_score(
    dmat_true,
    dmat_pred,
    all_atom_mask,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
):
    n = all_atom_mask.shape[-2]
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))
    return score

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    score = dist_map_to_score(
        dmat_true,
        dmat_pred,
        all_atom_mask,
        cutoff,
        eps,
        per_residue,
    )

    return score

