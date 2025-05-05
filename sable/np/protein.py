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

from typing import Any, Mapping, Sequence, Union

import numpy as np
import torch
from torch import Tensor

from opencomplex.np.protein import Protein, PDB_CHAIN_IDS
import opencomplex.np.residue_constants as rc

def design_to_protein(design: Sequence[str], metas: Mapping[str, Mapping[str, Any]], batch: Mapping[str, Union[Tensor, Mapping[str, Any]]], predicted_positions: Mapping[str, Mapping[str, Tensor]]={}) -> Protein:
    """
    Takes predictions on [MASK] residues as well as prepared PDB data to constructs a Protein object, basically modified from from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All non-standard atoms will be ignored.

    :param design: the predicted sequence to be applied in the protein
    :param metas: the table for meta information during antibody design processing
    :param batch: the input batch data for forward procedure of antibody design
    :param predicted_positions: the structure predicted during training
    """

    batch_size = next(iter(metas.items()))[1]["collated_shape"][0]
    total_length = sum(x["collated_shape"][1] for x in metas.values())
    h_pdb = batch["H_pdb"][0] # heavy chain alway exists, pick one for dtype
    aatype = np.full((batch_size, total_length), rc.restype_num, dtype=h_pdb["aatype"].dtype)
    atom_mask = np.zeros((batch_size, total_length, rc.atom_type_num), dtype=h_pdb["atom_mask"].dtype)
    atom_positions = np.zeros((batch_size, total_length, rc.atom_type_num, 3), dtype=h_pdb["atom_positions"].dtype)
    b_factors = np.zeros((batch_size, total_length, rc.atom_type_num), dtype=h_pdb["b_factors"].dtype)
    chain_index = np.full((batch_size, total_length), PDB_CHAIN_IDS.index("X"), dtype=h_pdb["residue_index"].dtype)
    asym_id = np.zeros((batch_size, total_length), dtype=h_pdb["asym_id"].dtype)
    residue_index = np.zeros((batch_size, total_length), dtype=h_pdb["residue_index"].dtype)
    acc_len, design_offset = 0, 0
    for (chain, meta) in sorted(metas.items()): # go through chains
        l = meta["collated_shape"][1]
        for i in range(batch_size):
            pdb = batch[chain + "_pdb"][i]
            sequence_length = meta["sequence_length"][i]
            aatype[i, acc_len + 1 : acc_len + sequence_length + 1] = pdb["aatype"]
            if meta["mask_count"][i] > 0:
                cdr = pdb["cdr"]
                masked_cdrs = meta["masked_cdrs"]
                for k in range(sequence_length):
                    if int(cdr[k]) in masked_cdrs:
                        aatype[i, acc_len + k + 1] = rc.restype_order.get(design[design_offset], rc.restype_num)
                        design_offset += 1
            if chain in predicted_positions:
                pd = predicted_positions[chain]
                atom_mask[i, acc_len : acc_len + l + 1] = pd["atom_mask"][i].detach().cpu().numpy()
                atom_positions[i, acc_len : acc_len + l + 1] = pd["atom_positions"][i].detach().cpu().numpy()
            else:
                atom_mask[i, acc_len + 1 : acc_len + sequence_length + 1] = pdb["atom_mask"]
                atom_positions[i, acc_len + 1 : acc_len + sequence_length + 1] = pdb["atom_positions"]
            b_factors[i, acc_len + 1 : acc_len + sequence_length + 1] = pdb["b_factors"]
            chain_index[i, acc_len + 1 : acc_len + sequence_length + 1] = np.vectorize(PDB_CHAIN_IDS.index)(pdb["chain_index"])
            asym_id[i, acc_len + 1 : acc_len + sequence_length + 1] = pdb["asym_id"]
            residue_index[i, acc_len + 1 : acc_len + sequence_length + 1] = pdb["residue_index"]
        acc_len += meta["collated_shape"][1]
    proteins = []
    for i in range(batch_size):
        proteins.append(
            Protein(
                atom_positions=atom_positions[i],
                atom_mask=atom_mask[i],
                aatype=aatype[i],
                residue_index=residue_index[i],
                asym_id=asym_id[i],
                chain_index=chain_index[i],
                b_factors=b_factors[i],
                parents=None,
                parents_chain_index=None,
            )
        )

    return proteins

def pack_asym_id(metas: Mapping[str, Mapping[str, Any]], batch: Mapping[str, Tensor]) -> Tensor:
    """
    Collect the asym_id from each chain and concatenate to return

    :param metas: the table for meta information during antibody design processing
    :param batch: the input for the forward procedure, will use the PDB information stored inside
    """

    batch_size = next(iter(metas.items()))[1]["collated_shape"][0]
    total_length = sum(x["collated_shape"][1] for x in metas.values())
    asym_id = torch.zeros((batch_size, total_length), dtype = torch.int32)
    acc_len = 0
    for (chain, meta) in sorted(metas.items()): # go through chains
        pdb = batch[chain + "_pdb"]
        for i in range(batch_size):
            asym_id[i, acc_len + 1 : acc_len + meta["sequence_length"][i] + 1] = torch.tensor(pdb[i]["asym_id"])
        acc_len += meta["collated_shape"][1]

    return asym_id

