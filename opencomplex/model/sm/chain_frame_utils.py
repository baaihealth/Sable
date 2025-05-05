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


def generate_avg_pool_matrix(chain_id, dtype):
    one_hot_matrix = torch.nn.functional.one_hot(chain_id.view(-1, chain_id.shape[-1]).long()).type(dtype)
    sum_pool_matrix = torch.bmm(one_hot_matrix, one_hot_matrix.transpose(-1, -2))
    avg_pool_matrix = sum_pool_matrix / sum_pool_matrix.sum(-1, keepdim=True)
    return avg_pool_matrix


def qt_vec_avg_pool(qt_vec, weights):
    q_vec, t_vec = qt_vec[..., :3], qt_vec[..., 3:]
    q_vec = torch.concat([torch.ones_like(q_vec)[..., :1], q_vec], dim=-1)
    q_vec = q_vec / q_vec.norm(dim=-1, keepdim=True)
    q_mat = torch.bmm(q_vec.view(-1, 4, 1), q_vec.view(-1, 1, 4)).view(-1, weights.shape[-1], 4, 4)  # [*, N, 4, 4]
    q_mat_noise = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]], dtype=q_mat.dtype, device=q_mat.device, requires_grad=False)
    q_mat = q_mat + q_mat_noise * 1e-6  # add noise to avoid same eigenvalues
    q_mat_avg_pooled = torch.bmm(q_mat.view(-1, weights.shape[-1], 16).transpose(-1, -2), weights).transpose(-1, -2).view_as(q_mat)
    q_vec_avg_pooled = torch.linalg.eigh(q_mat_avg_pooled.view(-1, 4, 4))[1][..., -1].view_as(q_vec)
    q_vec_avg_pooled = (q_vec_avg_pooled / q_vec_avg_pooled[..., :1].tile((4,)))[..., 1:]
    t_vec_avg_pooled = torch.bmm(t_vec.view(-1, weights.shape[-1], 3).transpose(-1, -2), weights).transpose(-1, -2).view_as(t_vec)
    return torch.concat([q_vec_avg_pooled, t_vec_avg_pooled], dim=-1)


def points_avg_pool(points, weights):
    return torch.einsum('...iab,ij->...jab', points, weights)
