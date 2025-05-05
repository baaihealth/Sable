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

from typing import Mapping

import numpy as np

from opencomplex.np import nucleotide_constants, residue_constants, protein
from opencomplex.utils.complex_utils import ComplexType


FeatureDict = Mapping[str, np.ndarray]


def empty_template_feats(n_res) -> FeatureDict:
    return {
        "template_butype": np.zeros((0, n_res)).astype(np.int64),
        "template_all_atom_positions": np.zeros((0, n_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, n_res, 37)).astype(np.float32),
    }


def make_sequence_features(
    sequence: str,
    description: str,
    num_bu: int,
    chain_type=ComplexType.PROTEIN,
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    constants = (
        residue_constants if chain_type == ComplexType.PROTEIN else nucleotide_constants
    )
    features["butype"] = constants.sequence_to_onehot(
        sequence=sequence,
        mapping=constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_bu"] = np.zeros((num_bu,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
    features["residue_index"] = np.array(range(num_bu), dtype=np.int32)
    features["seq_length"] = np.array([num_bu] * num_bu, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return features


def _aatype_to_str_sequence(aatype):
    return "".join(
        [residue_constants.restypes_with_x[aatype[i]] for i in range(len(aatype))]
    )


def make_protein_features(
    protein_object: protein.Protein,
    description: str,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_bu=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.0]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(1.0 if _is_distillation else 0.0).astype(
        np.float32
    )
    pdb_feats["chain_index"] = protein_object.chain_index

    return pdb_feats


def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.0,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if is_distillation:
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats

