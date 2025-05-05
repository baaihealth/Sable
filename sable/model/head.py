# Copyright 2025 Beijing Academy of Artificial Intelligence (BAAI)
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

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sable.util.tensor_util import get_activation_fn


class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        activation_fn: str,
        hidden: int=None,
    ) -> None:
        super().__init__()

        hidden = input_dim if not(hidden) else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation = get_activation_fn(activation_fn)

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        activation_fn: str,
        weight: Optional[Tensor]=None,
    ) -> None:
        super().__init__()

        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation = get_activation_fn(activation_fn)
        self.layer_norm = nn.LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        features: Tensor,
        masked_tokens: Optional[Tensor]=None,
        **kwargs
    ) -> Tensor:
        # Only project the masked tokens while training,
        # saves both memory and computation
        if not(masked_tokens is None):
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads: int,
        atom_level: int,
        activation_fn: str,
    ) -> None:
        super().__init__()

        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation = get_activation_fn(activation_fn)
        self.parameter = nn.Parameter(torch.ones(1, 1, atom_level, 1, atom_level, heads))
        self.atom_level = atom_level

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        bsz, seq_len, _, _ = x.size()
        x = x.unsqueeze(2)
        x = x.unsqueeze(-2)
        x = torch.mul(x, self.parameter)
        x = x.view(bsz, seq_len * self.atom_level, seq_len * self.atom_level, -1)
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len * self.atom_level, seq_len * self.atom_level)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x

