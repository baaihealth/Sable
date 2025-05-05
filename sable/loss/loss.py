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

from Bio.PDB import *
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Structure import Structure
from omegaconf import DictConfig
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Any, Mapping, Sequence, Tuple
import warnings

from opencomplex.np.protein import Protein, to_pdb
from opencomplex.loss.loss import fape_loss

from sable.np import residue_constant as rc
from sable.np.protein import design_to_protein, pack_asym_id


def tocdr(resseq: int) -> int:
    """
    Identify the CDR for a residue's sequence number

    :param resseq: The sequence number (index) for the residue in the protein
    """

    # map the residue sequence number to related CDR value, 0 for residues not in CDR
    if 27 <= resseq <= 38:
        return 1
    elif 56 <= resseq <= 65:
        return 2
    elif 105 <= resseq <= 117:
        return 3
    else:
        return 0


def parse_structure(protein: Protein, id: str) -> Structure:
    """
    Parse the `Protein` structure to `Structure` in Biopython for further manipulation

    :param protein: The `Protein` structure to be used producing lines in PDB file, aligning outputs to existing solution
    :param id: The PDB ID for that protein
    """

    # The following implementation simplifies the `get_structure` method of `PDBParser` from biopython/Bio/PDB/PDBParser.py
    parser = PDBParser() # for ground truth ones
    warnings.filterwarnings("ignore", category=PDBConstructionWarning)
    parser.header = None
    parser.trailer = None
    parser.structure_builder.init_structure(id)
    lines = to_pdb(protein).split("\n")
    parser._parse(lines)
    parser.structure_builder.set_header(parser.header)
    return parser.structure_builder.get_structure()[0]


class SableLoss(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for loss, majorly the `loss_distance_scalar` and `loss_residue_scalar` from network configuration
        """

        super(SableLoss, self).__init__()

        self.atom_level = config.atom_level
        self.dist_mean = 6.312581655060595
        self.dist_std = 3.3899264663911888
        self.loss_distance_scalar = config.loss_distance_scalar
        self.loss_residue_scalar = config.loss_residue_scalar

    def forward(self, output: Tuple[Tensor, Tensor], batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        :param output: The output from the `forward` method of the model
        :param batch: The batch for the loss
        """

        target = batch["tgt_residue"]
        batch_size = target.size(0)
        residue_mask = target.ne(rc.PAD)
        count = residue_mask.long().sum().to(torch.float32)
        logits_encoder, encoder_distance = output
        if residue_mask is not None:
            target = target[residue_mask]
        residue_loss = F.nll_loss(
            F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
            target,
            ignore_index=rc.PAD,
            reduction="mean",
        )
        prediction = logits_encoder.argmax(dim=-1)
        hit = (prediction == target).long().sum().to(torch.float32)
        loss = residue_loss * self.loss_residue_scalar
        logging_output = {
            "residue_loss": residue_loss,
            "hit": hit,
            "count": count,
        }

        if encoder_distance is not None:
            distance_mask = torch.repeat_interleave(residue_mask, self.atom_level).view(batch_size, -1)
            distance_loss = self.calculate_distance_loss(encoder_distance, distance_mask, batch["tgt_distance"], normalize=False)
            loss = loss + distance_loss * self.loss_distance_scalar
            logging_output["distance_loss"] = distance_loss.data

        logging_output["loss"] = loss
        return logging_output

    def calculate_distance_loss(self, distance: Tensor, distance_mask: Tensor, tgt_distance: Tensor, normalize: bool=False) -> Tensor:
        """
        Calculate the distance loss (L1) for the masked part according to `distance_mask`

        :param distance: The predicted distance tensor
        :param distance_mask: The distance mask which comes from the residue mask
        :param tgt_distance: The ground truth distance tensor
        :param normalize: Indicator for whether to make the distance with zero mean and standard deviation as one
        """

        masked_distance = distance[distance_mask, :]
        masked_distance_target = tgt_distance[distance_mask]
        non_pad_pos = masked_distance_target > 0
        if normalize:
            masked_distance_target = (
                masked_distance_target.float() - self.dist_mean
            ) / self.dist_std
        distance_loss = F.smooth_l1_loss(
            masked_distance[non_pad_pos].view(-1).float(),
            masked_distance_target[non_pad_pos].view(-1),
            reduction="mean",
            beta=1.0,
        )
        return distance_loss


class ProteinDesignLoss(nn.Module):
    def __init__(self):
        super(ProteinDesignLoss, self).__init__()

    def forward(self, output: Tensor, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        :param output: The output from the `forward` method of the model
        :param batch: The batch for the loss
        """

        target = batch["tgt_residue"]
        residue_mask = target.ne(rc.PAD)
        if residue_mask is not None:
            target = target[residue_mask]
        loss = F.nll_loss(
            F.log_softmax(output, dim=-1, dtype=torch.float32),
            target,
            ignore_index=rc.PAD,
            reduction="mean",
        )
        prediction = output.argmax(dim=-1)
        hit = (prediction == target).long().sum()
        count = residue_mask.long().sum()

        return {
            "loss": loss,
            "hit": hit,
            "count": count,
        }


class AntibodyDesignLoss(nn.Module):
    def __init__(self, config: DictConfig):
        """
        :param config: Settings for loss, majorly for the configuration of FAPE loss
        """

        super(AntibodyDesignLoss, self).__init__()

        self.atom_level = config.atom_level
        self.config = config

    def forward(self, output: Tuple[Mapping[str, Tensor], Mapping[str, Tensor], Mapping[str, Any], Mapping[str, Tensor], Mapping[str, Tensor]], batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        :param output: The output from the `forward` method of the model, all are dictionaries for different CDRs
        :param batch: The batch for the loss
        """

        logits, distance_predict, metas, features, outputs = output
        residue_loss, distance_loss, hit, count = 0, 0, 0, 0
        detailed_hits, detailed_counts, logging_output = {}, {}, {}
        acc_len = 0
        for chain in sorted(metas):
            if not(chain in logits):
                acc_len += metas[chain]["collated_shape"][1]
                continue
            target = batch[chain + "_tgt_residue"] # for residue loss
            residue_mask = target.ne(rc.PAD)
            target = target[residue_mask] - rc.restype_begin
            tmp = F.nll_loss(
                F.log_softmax(logits[chain], dim=-1, dtype=torch.float32),
                target,
                reduction="mean"
            )
            residue_loss += tmp

            prediction = logits[chain].argmax(dim=-1) # for accuracy
            correctness = (prediction == target)
            masked_cdrs = metas[chain]["masked_cdrs"]
            all_cdr = torch.cat([x["cdr"] for x in batch[chain + "_pdb"]])
            pred_in_cdr = all_cdr[torch.any(torch.stack([all_cdr == x for x in masked_cdrs], dim=0), dim=0)]
            for cdr in masked_cdrs:
                cdr_section = (pred_in_cdr == cdr)
                logging_output[chain + str(cdr) + "_hit"] = correctness[cdr_section].long().sum()
                logging_output[chain + str(cdr) + "_count"] = cdr_section.long().sum()
            hit += correctness.long().sum()
            count += len(target)

            tgt_distance = batch[chain + "_tgt_distance"] # for distance loss
            distance_mask = torch.zeros(tgt_distance.shape, dtype=torch.bool)
            sequence_length = metas[chain]["sequence_length"]
            for (i, l) in enumerate(sequence_length):
                dl = self.atom_level * l
                distance_mask[i, self.atom_level : dl + self.atom_level, self.atom_level : dl + self.atom_level] = torch.ones((dl, dl), dtype=torch.bool).fill_diagonal_(False)
            tmp = F.smooth_l1_loss(
                distance_predict[chain][distance_mask].float(),
                tgt_distance[distance_mask].float(),
                reduction="mean",
                beta=1.0,
            )
            logging_output[chain + "_distance_loss"] = tmp
            distance_loss += tmp

            self.calculate_rmsd(prediction, chain, metas[chain], acc_len, outputs, batch, logging_output)
            acc_len += metas[chain]["collated_shape"][1]

        asym_id = pack_asym_id(metas, batch).to(outputs["single"].device) # the fape loss or structure loss
        oc_fape_loss, _ = fape_loss(outputs, features, self.config.fape, asym_id=asym_id)
        if torch.isnan(oc_fape_loss) or torch.isinf(oc_fape_loss):
            oc_fape_loss = oc_fape_loss.new_tensor(0.0, required_grad=True)
        structure_loss = self.config.fape.weight * oc_fape_loss * sum(x["sequence_length"] for x in metas.values()).sqrt().mean() # apply the trick from AlphaFold2 but skip the cropping here since no `crop_to_size` is applied

        loss = residue_loss + distance_loss + structure_loss

        logging_output.update({
            "loss": loss,
            "distance_loss": distance_loss,
            "residue_loss": residue_loss,
            "structure_loss": structure_loss,
            "hit": hit,
            "count": count,
        })

        return logging_output

    def calculate_rmsd(self, prediction: Sequence[int], chain: str, meta: Mapping[str, Any], offset: int, outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor], logging_output: Mapping[str, Tensor]) -> None:
        """
        Calculate the RMSD for a given chain, and keep in `logging_output`

        :param prediction: The predicted residue IDs
        :param chain: The chain ID, with value as "H", "L", or "Ag"
        :param meta: The meta information for different chains, is majorily used for transferring intermediate data
        :param outputs: The outputs from the model, including those from the structure module
        :param batch: The batch that was passed in loss
        :param logging_output: The value to return for the loss, add RMSD values to it
        """

        gt_meta = meta.copy() # for ground truth proteins
        gt_meta["masked_cdrs"] = {}
        gt_meta["mask_count"] = torch.zeros_like(gt_meta["mask_count"])
        gt_proteins = design_to_protein([], {chain: gt_meta}, batch)
        design = [rc.residue_type_explain(int(x) + rc.restype_begin) for x in prediction] # for predicted proteins
        pred_proteins = design_to_protein(
            design,
            { chain: meta },
            batch,
            {
                chain: {
                    "atom_mask": outputs["final_atom_mask"][ : , offset : offset + meta["collated_shape"][1]],
                    "atom_positions": outputs["final_atom_positions"][ : , offset : offset + meta["collated_shape"][1]],
                }
            })

        masked_cdrs = meta["masked_cdrs"]
        RMSDs = {}
        for (gt_protein, pred_protein) in zip(gt_proteins, pred_proteins):
            residues = {}
            pred_structure = parse_structure(pred_protein, "pred") # for predicted one
            entities = Selection.unfold_entities(pred_structure, "R")
            for entity in entities:
                cdr = tocdr(entity.get_full_id()[3][1])
                if cdr in masked_cdrs:
                    if not(cdr in residues):
                        residues[cdr] = ([], [])
                    residues[cdr][0].append(entity)
                chainid = entity.get_full_id()[2]

            gt_structure = parse_structure(gt_protein, "native")[chainid] # for ground truth one
            entities = Selection.unfold_entities(gt_structure, "R")
            for entity in entities:
                cdr = tocdr(entity.get_full_id()[3][1])
                if cdr in masked_cdrs:
                    residues[cdr][1].append(entity)

            sup = Superimposer()
            for (cdr, residue_pairs) in residues.items():
                gt_atoms, pred_atoms = [], []
                for pred_residue, gt_residue in zip(residue_pairs[0], residue_pairs[1]):
                    gt_atom = Selection.unfold_entities(gt_residue, "A")
                    pred_atom = Selection.unfold_entities(pred_residue, "A")
                    gt_atom.sort()
                    pred_atom.sort()
                    gt_atom_id = [x.get_name() for x in gt_atom]
                    pred_atom_id = [x.get_name() for x in pred_atom]
                    if gt_atom_id != pred_atom_id:
                        atom_id = list(set(gt_atom_id) & set(pred_atom_id))
                        gt_atom = [x for x in gt_atom if x.get_name() in atom_id]
                        pred_atom = [x for x in pred_atom if x.get_name() in atom_id]
                    gt_atoms += gt_atom
                    pred_atoms += pred_atom
                sup.set_atoms(gt_atoms, pred_atoms)
                RMSDs[cdr] = RMSDs.get(cdr, 0) + sup.rms

        for (cdr, RMSD) in RMSDs.items():
            logging_output[chain + str(cdr) + "_RMSD"] = RMSD / len(gt_proteins)


class ModelQualityAssessmentLoss(nn.Module):
    def __init__(self):
        super(ModelQualityAssessmentLoss, self).__init__()

    def forward(self, output: Tensor, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        :param output: The output from the `forward` method of the model
        :param batch: The batch for the loss
        """

        pred_lddt, pred_gdt_ts = output
        tgt_lddt, tgt_gdt_ts = batch["tgt_lddt"], batch["tgt_gdt_ts"]

        lddt_mask = batch["src_residue"].ne(rc.PAD)

        pred_lddt = pred_lddt[lddt_mask]
        tgt_lddt = tgt_lddt[lddt_mask]

        gdt_ts_loss = F.mse_loss(pred_gdt_ts.float(), tgt_gdt_ts.float(), reduction="mean")
        lddt_loss = F.mse_loss(pred_lddt.float(), tgt_lddt.float(), reduction="mean")

        loss = 100 * (gdt_ts_loss + lddt_loss)

        logging_output = {
            "loss": loss,
            "gdt_ts_loss": gdt_ts_loss,
            "lddt_loss": lddt_loss,
        }
        if not(self.training):
            logging_output.update({
                "id": batch["id"],
                "pred_gdt_ts": pred_gdt_ts,
                "tgt_gdt_ts": tgt_gdt_ts,
                "pred_lddt": pred_lddt,
                "tgt_lddt": tgt_lddt,
            })

        return logging_output


class BindingAffinityLoss(nn.Module):
    def __init__(self):
        super(BindingAffinityLoss, self).__init__()

    def forward(self, output: Tensor, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        :param output: The output from the `forward` method of the model
        :param batch: The batch for the loss
        """

        pred_ddg = output
        tgt_ddg = batch["tgt_ddg"]

        loss = F.mse_loss(pred_ddg.float(), tgt_ddg.float(), reduction="mean")

        return {
            "loss": loss,
            "pred_ddg": pred_ddg,
            "tgt_ddg": tgt_ddg,
        }


class EnzymeCatalyzedReactionClassificationLoss(nn.Module):
    def __init__(self):
        super(EnzymeCatalyzedReactionClassificationLoss, self).__init__()

    def forward(self, output: Tensor, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        :param output: The output from the `forward` method of the model
        :param batch: The batch for the loss
        """

        logits = output
        tgt_class = batch["tgt_class"]

        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            tgt_class,
            reduction="mean",
        )
        pred_class = logits.argmax(dim=-1)

        return {
            "loss": loss,
            "pred_class": pred_class,
            "tgt_class": tgt_class,
        }


class FoldClassificationLoss(nn.Module):
    def __init__(self):
        super(FoldClassificationLoss, self).__init__()

    def forward(self, output: Tensor, batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """
        :param output: The output from the `forward` method of the model
        :param batch: The batch for the loss
        """

        logits = output
        tgt_class = batch["tgt_class"]

        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            tgt_class,
            reduction="mean",
        )
        pred_class = logits.argmax(dim=-1)

        return {
            "loss": loss,
            "pred_class": pred_class,
            "tgt_class": tgt_class,
        }

