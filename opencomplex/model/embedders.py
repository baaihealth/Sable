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
import torch.nn as nn
import torch.nn.functional as F

from opencomplex.model.primitives import Linear
from opencomplex.data.data_transforms import make_one_hot


class RelPosEncoder(nn.Module):
    def __init__(
        self,
        c_z,
        max_relative_idx,
        max_relative_chain=0,
        use_chain_relative=False,
    ):
        super(RelPosEncoder, self).__init__()
        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain
        self.use_chain_relative = use_chain_relative
        self.no_bins = 2 * max_relative_idx + 1
        if max_relative_chain > 0:
            self.no_bins += 2 * max_relative_chain + 2
        if use_chain_relative:
            self.no_bins += 1

        self.linear_relpos = Linear(self.no_bins, c_z)

    def forward(self, residue_index, asym_id=None, sym_id=None, entity_id=None):
        d = residue_index[..., None] - residue_index[..., None, :]

        if asym_id is None:
            boundaries = torch.arange(
                start=-self.max_relative_idx, end=self.max_relative_idx + 1, device=d.device
            )
            reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
            d = d[..., None] - reshaped_bins
            d = torch.abs(d)
            d = torch.argmin(d, dim=-1)
            d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
            d = d.to(residue_index.dtype)
            rel_feat = d
        else:
            rel_feats = []
            asym_id_same = torch.eq(asym_id[..., None], asym_id[..., None, :])
            offset = residue_index[..., None] - residue_index[..., None, :]

            clipped_offset = torch.clamp(
                offset + self.max_relative_idx, 0, 2 * self.max_relative_idx - 1)

            final_offset = torch.where(asym_id_same, clipped_offset,
                                    (2 * self.max_relative_idx) *
                                    torch.ones_like(clipped_offset))

            rel_pos = make_one_hot(final_offset, 2 * self.max_relative_idx + 1)
            rel_feats.append(rel_pos)

            if self.use_chain_relative:
                entity_id_same = torch.eq(entity_id[..., None], entity_id[..., None, :])
                rel_feats.append(entity_id_same.type(rel_pos.dtype)[..., None])

            if self.max_relative_chain > 0:
                rel_sym_id = sym_id[..., None] - sym_id[..., None, :]
                max_rel_chain = self.max_relative_chain

                clipped_rel_chain = torch.clamp(
                    rel_sym_id + max_rel_chain, 0, 2 * max_rel_chain)

                final_rel_chain = torch.where(entity_id_same, clipped_rel_chain,
                                            (2 * max_rel_chain + 1) *
                                            torch.ones_like(clipped_rel_chain))
                rel_chain = make_one_hot(final_rel_chain, 2 * self.max_relative_chain + 2)
                rel_feats.append(rel_chain)

            rel_feat = torch.cat(rel_feats, dim=-1)

        return self.linear_relpos(rel_feat)

