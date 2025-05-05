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

from omegaconf import DictConfig
import torch
from torch import Tensor, nn
from typing import Any, Mapping, Tuple

from opencomplex.model.sm.structure_module_protein import StructureModuleProtein
from opencomplex.utils.feats import dense_atom_to_all_atom

from sable.data.data_pipeline import design_to_feature
from sable.model.embedder import DistanceEncoder, TransformerEncoderWithPair
from sable.model.head import DistanceHead, MaskLMHead
from sable.np.protein import design_to_protein
from sable.np import residue_constant as rc


class Sable(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for components in the network, including dimensions, dropouts, etc.
        """

        super(Sable, self).__init__()

        self.token_encoder = nn.Embedding(rc.dictionary_size, config.encoder_embed_dim, padding_idx=rc.PAD)
        self.distance_encoder = DistanceEncoder(config.encoder_attention_heads, config.activation_fn, config.atom_level)
        self.pair_encoder = TransformerEncoderWithPair(
            encoder_layers=config.encoder_layers,
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            attention_heads=config.encoder_attention_heads,
            emb_dropout=config.emb_dropout,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            activation_fn=config.activation_fn,
            post_ln=config.post_ln,
        )
        if config.loss_residue_scalar > 0:
            self.lm_head = MaskLMHead(embed_dim=config.encoder_embed_dim, output_dim=rc.dictionary_size, activation_fn=config.activation_fn, weight=None)
        if config.loss_distance_scalar > 0:
            self.dist_head = DistanceHead(config.encoder_attention_heads, config.atom_level, config.activation_fn)

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        :param batch: The batch for the forward pass
        """

        (src_coordinate, src_distance, src_residue, tgt_residue) = (batch["src_coordinate"], batch["src_distance"], batch["src_residue"], batch["tgt_residue"]) # extract the inputs out of the `batch` dictionary

        padding_mask = src_residue.eq(rc.PAD)
        if not(padding_mask.any()):
            padding_mask = None

        x = self.token_encoder(src_residue)
        graph_attn_bias = self.distance_encoder(src_coordinate, src_distance)

        encoder_rep, encoder_pair_rep = self.pair_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        logits, encoder_distance = None, None
        if hasattr(self, "lm_head"):
            logits = self.lm_head(encoder_rep, tgt_residue.ne(rc.PAD))
        if hasattr(self, "dist_head"):
            encoder_distance = self.dist_head(encoder_pair_rep)

        return logits, encoder_distance


class ProteinDesign(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for components in the sable network, including dimensions, dropouts, etc.
        """

        super(ProteinDesign, self).__init__()

        self.sable = Sable(config.sable)
        self.seq_design = MaskLMHead(embed_dim=config.sable.encoder_embed_dim, output_dim=rc.dictionary_size, activation_fn=config.sable.activation_fn, weight=None)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        """
        :param batch: The batch for the forward pass
        """

        (src_coordinate, src_distance, src_residue, tgt_residue) = (batch["src_coordinate"], batch["src_distance"], batch["src_residue"], batch["tgt_residue"]) # extract the inputs out of the `batch` dictionary

        padding_mask = src_residue.eq(rc.PAD)
        x = self.sable.token_encoder(src_residue)
        graph_attn_bias = self.sable.distance_encoder(src_coordinate, src_distance)
        encoder_rep, _ = self.sable.pair_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        logits = self.seq_design(encoder_rep, tgt_residue.ne(rc.PAD))

        return logits


class AntibodyDesign(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for components in the sable network, including dimensions, dropouts, etc.
        """

        super(AntibodyDesign, self).__init__()

        self.cdr_mask = config.cdr_mask
        self.recycling = config.recycling
        self.sable = Sable(config.sable)
        if self.recycling:
            self.concat_decoder = TransformerEncoderWithPair(
                encoder_layers=config.concat.encoder_layers,
                embed_dim=config.concat.encoder_embed_dim,
                ffn_embed_dim=config.concat.encoder_ffn_embed_dim,
                attention_heads=config.concat.encoder_attention_heads,
                emb_dropout=config.concat.emb_dropout,
                dropout=config.concat.dropout,
                attention_dropout=config.concat.attention_dropout,
                activation_dropout=config.concat.activation_dropout,
                activation_fn=config.concat.activation_fn,
                post_ln=config.concat.post_ln,
            )
        self.seq_design = MaskLMHead(
            embed_dim=config.sable.encoder_embed_dim + config.sable.encoder_attention_heads * 2,
            output_dim=rc.restype_num,
            activation_fn=config.sable.activation_fn,
            weight=None,
        )
        self.distance_project = DistanceHead(
            config.sable.encoder_embed_dim + config.sable.encoder_attention_heads,
            config.sable.atom_level,
            config.sable.activation_fn,
        )
        self.single_linear = nn.Linear(in_features=config.sable.encoder_embed_dim, out_features=config.structure_module.c_s)
        self.pair_linear = nn.Linear(in_features=config.sable.encoder_attention_heads, out_features=config.structure_module.c_z)
        self.sm = StructureModuleProtein(**config.structure_module)

    def forward(self, batch: Mapping[str, Any]) -> Tuple[Mapping[str, Tensor], Mapping[str, Tensor], Mapping[str, Any], Mapping[str, Tensor], Mapping[str, Tensor]]:
        """
        :param batch: The batch for the forward pass
        """

        metas, padding_masks, residue_masks, encoder_reps, encoder_pair_reps = {}, {}, {}, {}, {}
        for (chain, masked_cdrs) in self.cdr_mask.items():
            (src_coordinate, src_distance, src_residue) = (batch[chain + "_src_coordinate"], batch[chain + "_src_distance"], batch[chain + "_src_residue"]) # extract the inputs out of the `batch` dictionary

            padding_mask = src_residue.eq(rc.PAD)
            padding_masks[chain] = padding_mask
            residue_masks[chain] = src_residue.eq(rc.MASK)
            metas[chain] = { # add one object for storing metas to avoid duplicated computation
                "masked_cdrs": masked_cdrs,
                "collated_shape": src_residue.shape,
                "sequence_length": src_residue.size(1) - padding_mask.long().sum(dim=1) - 2,
                "mask_count": residue_masks[chain].long().sum(dim=1),
            }

            x = self.sable.token_encoder(src_residue)
            graph_attn_bias = self.sable.distance_encoder(src_coordinate, src_distance)
            encoder_rep, encoder_pair_rep = self.sable.pair_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
            encoder_reps[chain] = encoder_rep
            encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0
            encoder_pair_reps[chain] = encoder_pair_rep

        # prepare for recycling through concatenation, and the pair representations are concatenated on diagonal
        concat_reps = torch.cat([x[1] for x in sorted(encoder_reps.items())], dim=-2)
        concat_masks = torch.cat([x[1] for x in sorted(padding_masks.items())], dim=-1)
        concat_attn_biases = torch.zeros(concat_reps.size(0), concat_reps.size(1), concat_reps.size(1), next(iter(encoder_pair_reps.items()))[1].size(-1)).type_as(concat_reps)
        acc_len = 0
        for (chain, encoder_pair_rep) in sorted(encoder_pair_reps.items()):
            seq_len = encoder_pair_rep.size(1)
            new_acc_len = acc_len + seq_len
            concat_attn_biases[ : , acc_len : new_acc_len, acc_len : new_acc_len, : ] = encoder_pair_rep
            acc_len = new_acc_len

        for _ in range(self.recycling): # recycling, here assign all -inf to zero to avoid NaN in fape loss
            concat_reps, concat_attn_biases = self.concat_decoder(concat_reps, padding_mask=concat_masks, attn_mask=concat_attn_biases.permute(0, 3, 1, 2).reshape(-1, acc_len, acc_len).contiguous())
            concat_attn_biases[concat_attn_biases == float("-inf")] = 0

        logits, distance_predict = {}, {}
        acc_len = 0
        for (chain, meta) in sorted(metas.items()):
            collated_length = meta["collated_shape"][1]
            new_acc_len = acc_len + collated_length

            if meta["mask_count"].sum() > 0:
                encoder_rep = concat_reps[ : , acc_len : new_acc_len]
                encoder_pair_rep = concat_attn_biases[ : , acc_len : new_acc_len, acc_len : new_acc_len, : ]

                pair_rep1 = torch.sum(encoder_pair_rep, dim=1) / collated_length
                pair_rep2 = torch.sum(encoder_pair_rep, dim=2) / collated_length
                decoder_rep = torch.cat([encoder_rep, pair_rep1, pair_rep2], dim=-1)

                logits[chain] = self.seq_design(decoder_rep, residue_masks[chain])

                decoder_pair_rep = torch.cat([encoder_pair_rep, encoder_rep.unsqueeze(-2).repeat(1, 1, collated_length, 1)], dim=-1)
                distance_predict[chain] = self.distance_project(decoder_pair_rep)

            acc_len = new_acc_len

        masked_pred = torch.cat([x[1] for x in sorted(logits.items())], dim=0).argmax(dim=-1)
        design = [rc.residue_type_explain(int(x)) for x in masked_pred]

        proteins = design_to_protein(design, metas, batch)
        feats = design_to_feature(proteins, "protein", rc.restype_num)
        features = {}
        for k, v in feats.items():
            features[k] = v.to(src_residue.device)
        outputs = {}
        outputs["single"] = self.single_linear(concat_reps)
        outputs["pair"] = self.pair_linear(concat_attn_biases)

        outputs["sm"] = self.sm(outputs, features["butype"], mask=features["seq_mask"], inplace_safe=False, _offload_inference=False)
        outputs["final_atom_positions"] = dense_atom_to_all_atom(outputs["sm"]["positions"][-1], features)
        outputs["final_atom_mask"] = features["all_atom_exists"]

        return logits, distance_predict, metas, features, outputs


class ModelQualityAssessment(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for components in the sable network, including dimensions, dropouts, etc.
        """

        super(ModelQualityAssessment, self).__init__()

        self.sable = Sable(config.sable)
        feat_dim = config.sable.encoder_embed_dim + config.sable.encoder_attention_heads * 2
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.lddt_project = nn.Linear(feat_dim, 1, bias=True)
        self.gdt_ts_project = nn.Linear(feat_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        :param batch: The batch for the forward pass
        """

        (src_coordinate, src_distance, src_residue) = (batch["src_coordinate"], batch["src_distance"], batch["src_residue"]) # extract the inputs out of the `batch` dictionary

        padding_mask = src_residue.eq(rc.PAD)
        x = self.sable.token_encoder(src_residue)
        graph_attn_bias = self.sable.distance_encoder(src_coordinate, src_distance)
        encoder_rep, encoder_pair_rep = self.sable.pair_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        collated_length = src_residue.size(1)
        pair_rep1 = torch.sum(encoder_pair_rep, dim=1) / collated_length
        pair_rep2 = torch.sum(encoder_pair_rep, dim=2) / collated_length
        decoder_rep = torch.cat([encoder_rep, pair_rep1, pair_rep2], dim=-1)

        decoder_rep = self.mlp(decoder_rep)
        pred_lddt = self.lddt_project(decoder_rep).squeeze(-1)
        pred_lddt = self.sigmoid(pred_lddt)

        pred_gdt_ts = self.gdt_ts_project(decoder_rep).squeeze(-1)
        pred_gdt_ts = pred_gdt_ts * (src_residue.ne(rc.PAD))
        pred_gdt_ts = torch.mean(pred_gdt_ts, dim=1)
        pred_gdt_ts = self.sigmoid(pred_gdt_ts)

        return pred_lddt, pred_gdt_ts


class BindingAffinity(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for components in the sable network, including dimensions, dropouts, etc.
        """

        super(BindingAffinity, self).__init__()

        self.sable = Sable(config.sable)
        self.ddg_pred = MaskLMHead(embed_dim=config.sable.encoder_embed_dim * 2, output_dim=1, activation_fn=config.sable.activation_fn, weight=None)

    def forward(self, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        :param batch: The batch for the forward pass
        """

        encoder_reps = dict()
        for t in ["wild", "mutant"]:
            (src_coordinate, src_distance, src_residue) = (batch["src_" + t + "_coordinate"], batch["src_" + t + "_distance"], batch["src_" + t + "_residue"]) # extract the inputs out of the `batch` dictionary

            padding_mask = src_residue.eq(rc.PAD)
            x = self.sable.token_encoder(src_residue)
            graph_attn_bias = self.sable.distance_encoder(src_coordinate, src_distance)
            encoder_reps[t] = self.sable.pair_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)[0]

        feat_wm = torch.cat([encoder_reps["wild"], encoder_reps["mutant"]], dim=-1)
        feat_mw = torch.cat([encoder_reps["mutant"], encoder_reps["wild"]], dim=-1)

        per_residue_ddg = self.ddg_pred(feat_wm) - self.ddg_pred(feat_mw)

        per_residue_ddg = per_residue_ddg.squeeze(-1)

        mask = batch["src_wild_residue"].ne(rc.PAD)
        per_residue_ddg = per_residue_ddg * mask
        ddg = per_residue_ddg.sum(dim=1)

        return ddg


class EnzymeCatalyzedReactionClassification(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for components in the network, including dimensions, dropouts, etc.
        """

        super(EnzymeCatalyzedReactionClassification, self).__init__()

        self.sable = Sable(config.sable)
        self.lm_head = MaskLMHead(embed_dim=config.sable.encoder_embed_dim, output_dim=384, activation_fn=config.sable.activation_fn, weight=None)

    def forward(self, batch):
        """
        :param batch: The batch for the forward pass
        """

        (src_coordinate, src_distance, src_residue) = (batch["src_coordinate"], batch["src_distance"], batch["src_residue"]) # extract the inputs out of the `batch` dictionary

        padding_mask = src_residue.eq(rc.PAD)
        x = self.sable.token_encoder(src_residue)
        graph_attn_bias = self.sable.distance_encoder(src_coordinate, src_distance)
        encoder_rep = self.sable.pair_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)[0]

        logits = self.lm_head(encoder_rep)
        logits = logits.mean(dim=1)

        return logits


class FoldClassification(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for components in the network, including dimensions, dropouts, etc.
        """

        super(FoldClassification, self).__init__()

        self.sable = Sable(config.sable)
        self.lm_head = MaskLMHead(embed_dim=config.sable.encoder_embed_dim, output_dim=1195, activation_fn=config.sable.activation_fn, weight=None)

    def forward(self, batch):
        """
        :param batch: The batch for the forward pass
        """

        (src_coordinate, src_distance, src_residue) = (batch["src_coordinate"], batch["src_distance"], batch["src_residue"]) # extract the inputs out of the `batch` dictionary

        padding_mask = src_residue.eq(rc.PAD)
        x = self.sable.token_encoder(src_residue)
        graph_attn_bias = self.sable.distance_encoder(src_coordinate, src_distance)
        encoder_rep = self.sable.pair_encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)[0]

        logits = self.lm_head(encoder_rep)
        logits = logits.mean(dim=1)

        return logits

