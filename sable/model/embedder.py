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

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from opencomplex.model.embedders import RelPosEncoder
from opencomplex.model.primitives import Linear
import opencomplex.np.residue_constants as rc

from sable.model.head import NonLinearHead
from sable.util.tensor_util import get_activation_fn, softmax_dropout


class SpatialPosEncoder(nn.Module):
    def __init__(self, c_z, atom_level):
        super().__init__()

        self.min_bin = 3
        self.max_bin = 80
        self.no_bins = 128

        self.min_direct_bins = -50
        self.max_direct_bins = 50
        self.no_direct_bins = 128

        self.linear_relpos = Linear((self.no_bins + 1) + 3 * (self.no_direct_bins + 1), c_z)
        self.atom_level = atom_level


    def forward(self, all_atom_positions):
        ca_atom_positions = all_atom_positions[ : , rc.atom_order["CA"] : : self.atom_level, : ]

        ca_atom_dist = torch.pairwise_distance(ca_atom_positions[..., None, : ], ca_atom_positions[..., None, : , : ])

        ca_atom_dist = ca_atom_dist[..., None]
        lower = torch.linspace(self.min_bin, self.max_bin, self.no_bins, device=all_atom_positions.device)
        upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)

        dgram = ((ca_atom_dist > lower) * (ca_atom_dist < upper)).type(ca_atom_dist.dtype)

        n_atom_positions = all_atom_positions[ : , rc.atom_order["N"] : : self.atom_level, : ]
        c_atom_positions = all_atom_positions[ : , rc.atom_order["C"] : : self.atom_level, : ]

        ca_diff = ca_atom_positions[..., None, : ] - ca_atom_positions[..., None, : , : ]

        x_axis = F.normalize(n_atom_positions - ca_atom_positions, dim=-1)
        z_axis = torch.cross(x_axis, (c_atom_positions - ca_atom_positions), dim=-1)
        z_axis = F.normalize(z_axis, dim=-1)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)

        relative_pos = [
            torch.einsum("...ij, ...ij -> ...i", ca_diff, x_axis),
            torch.einsum("...ij, ...ij -> ...i", ca_diff, y_axis),
            torch.einsum("...ij, ...ij -> ...i", ca_diff, z_axis),
        ]
        relative_pos = torch.stack(relative_pos, dim=-1)[..., None]

        lower = torch.linspace(self.min_direct_bins, self.max_direct_bins, self.no_direct_bins - 1, device=all_atom_positions.device)
        lower = torch.cat([lower.new_tensor([-1e8]), lower[ : ]], dim=-1)
        upper = torch.cat([lower[1 : ], lower.new_tensor([1e8])], dim=-1)
        direction_gram = ((relative_pos > lower) * (relative_pos < upper)).type(ca_atom_dist.dtype)

        dgram = torch.argmax(dgram, dim=-1) + 1
        direction_gram = torch.argmax(direction_gram, dim=-1) + 1

        dgram = F.one_hot(dgram, num_classes=self.no_bins + 1)
        direction_gram = F.one_hot(direction_gram, num_classes=self.no_direct_bins + 1)

        rel_feat = torch.cat([
            dgram,
            direction_gram[..., 0, : ],
            direction_gram[..., 1, : ],
            direction_gram[..., 2, : ]], dim=-1).type(all_atom_positions.dtype)

        return self.linear_relpos(rel_feat)


class DistanceEncoder(nn.Module):
    def __init__(
        self,
        encoder_attention_heads: int,
        activation_fn: str,
        atom_level: int,
    ) -> None:
        super().__init__()

        self.atom_level = atom_level
        self.encoder_attention_heads = encoder_attention_heads
        self.rpe = RelPosEncoder(encoder_attention_heads, 32, 0, False)
        self.embed_distance1 = nn.Embedding(16, encoder_attention_heads)
        self.embed_distance2 = nn.Embedding(64, encoder_attention_heads)
        self.embed_distance3 = nn.Embedding(256, encoder_attention_heads)
        self.embed_distance4 = nn.Embedding(1024, encoder_attention_heads)
        self.embed_distance5 = nn.Embedding(4096, encoder_attention_heads)
        self.embed_distance6 = nn.Embedding(16384, encoder_attention_heads)
        self.gbf_proj = NonLinearHead(encoder_attention_heads, encoder_attention_heads, activation_fn)
        self.parameter = nn.Parameter(torch.ones(1, 1, atom_level, 1, atom_level, encoder_attention_heads))
        self.spatialposencoder = SpatialPosEncoder(encoder_attention_heads, atom_level)

    def forward(
        self,
        src_coordinate: Tensor,
        src_distance: Tensor,
    ) -> Tensor:
        bsz, n_node = src_distance.size(0), src_distance.size(-1)
        n_res = int(n_node / self.atom_level)

        coord_update = torch.zeros(src_coordinate.size(0), n_res, n_res, self.encoder_attention_heads).type_as(src_coordinate)
        for i in range(src_coordinate.size(0)):
            coord_update[i, ] = self.spatialposencoder(src_coordinate[i : i + 1, : ])

        position = torch.arange(0, n_res, dtype=torch.float).unsqueeze(0).repeat(bsz, 1).cuda()
        position_emb = self.rpe(position.type(src_distance.dtype)).squeeze()

        distance = torch.where(src_distance > 126, 127, src_distance)
        distance_bin = (distance / (128 / 16)).type(torch.int64)
        distance_embed = self.embed_distance1(distance_bin)
        distance_bin = (distance / (128 / 64)).type(torch.int64)
        distance_embed += self.embed_distance2(distance_bin)
        distance_bin = (distance / (128 / 256)).type(torch.int64)
        distance_embed += self.embed_distance3(distance_bin)
        distance_bin = (distance / (128 / 1024)).type(torch.int64)
        distance_embed += self.embed_distance4(distance_bin)
        distance_bin = (distance / (128 / 4096)).type(torch.int64)
        distance_embed += self.embed_distance5(distance_bin)
        distance_bin = (distance / (128 / 16384)).type(torch.int64)
        distance_embed += self.embed_distance6(distance_bin)
        gbf_feature = distance_embed
        gbf_feature = gbf_feature.view(bsz, n_res, self.atom_level, n_res, self.atom_level, -1)
        gbf_feature = torch.mul(gbf_feature, self.parameter)
        gbf_feature = torch.sum(gbf_feature, dim=2)
        gbf_feature = torch.sum(gbf_feature, dim=-2)
        gbf_result = self.gbf_proj(gbf_feature)

        graph_attn_bias = coord_update + position_emb + gbf_result
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_res, n_res)
        return graph_attn_bias


class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float=0.1,
        bias: bool=True,
        scaling_factor: float=1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
        key_padding_mask: Optional[Tensor]=None,
        attn_bias: Optional[Tensor]=None,
        return_attn: bool=False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim) * self.scaling
        if not(k is None):
            k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        if not(v is None):
            v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous().view(bsz * self.num_heads, -1, self.head_dim)

        assert not(k is None)
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if not(key_padding_mask is None) and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if not(key_padding_mask is None):
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if not(key_padding_mask is None):
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not(return_attn):
            attn = softmax_dropout(attn_weights, self.dropout, self.training, bias=attn_bias)
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(attn_weights, self.dropout, self.training, inplace=False)

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = o.view(bsz, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        o = self.out_proj(o)
        if not(return_attn):
            return o
        else:
            return o, attn_weights, attn

class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int=768,
        ffn_embed_dim: int=3072,
        attention_heads: int=8,
        dropout: float=0.1,
        attention_dropout: float=0.1,
        activation_dropout: float=0.0,
        activation_fn: str="gelu",
        post_ln=False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation = get_activation_fn(activation_fn)

        self.self_attn = SelfMultiheadAttention(embed_dim, attention_heads, dropout=attention_dropout)
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.post_ln = post_ln

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not(self.post_ln):
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(query=x, key_padding_mask=padding_mask, attn_bias=attn_bias, return_attn=return_attn)
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not(self.post_ln):
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not(return_attn):
            return x
        else:
            return x, attn_weights, attn_probs


class TransformerEncoderWithPair(nn.Module):
    def __init__(
        self,
        encoder_layers: int=6,
        embed_dim: int=768,
        ffn_embed_dim: int=3072,
        attention_heads: int=8,
        emb_dropout: float=0.1,
        dropout: float=0.1,
        attention_dropout: float=0.1,
        activation_dropout: float=0.0,
        activation_fn: str="gelu",
        post_ln: bool=False,
    ) -> None:

        super().__init__()

        self.emb_dropout = emb_dropout

        self.emb_layer_norm = nn.LayerNorm(embed_dim)
        if not(post_ln):
            self.final_layer_norm = nn.LayerNorm(embed_dim)
        else:
            self.final_layer_norm = None
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_embed_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                post_ln=post_ln,
            )
            for _ in range(encoder_layers)
        ])

    def forward(
        self,
        emb: Tensor,
        attn_mask: Optional[Tensor]=None,
        padding_mask: Optional[Tensor]=None,
    ) -> tuple[Tensor, Tensor]:

        bsz = emb.size(0)
        seq_len = emb.size(1)

        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)

        # account for padding while computing the representation
        if not(padding_mask is None):
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if not(attn_mask is None) and not(padding_mask is None):
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), fill_val)
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert not(attn_mask is None)
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)

        for i in range(len(self.layers)):
            x, attn_mask, _ = self.layers[i](x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True)

        if not(self.final_layer_norm is None):
            x = self.final_layer_norm(x)

        attn_mask = attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()

        return x, attn_mask

