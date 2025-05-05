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

import torch

from functools import partial

from opencomplex.data import data_transforms
from opencomplex.data.data_pipeline import make_pdb_features, FeatureDict
from opencomplex.data.feature_pipeline import np_to_tensor_dict
from opencomplex.np.protein import Protein
from opencomplex.utils.complex_utils import ComplexType
from opencomplex.utils.tensor_utils import dict_multimap

def process_tensors_without_msa_template(tensors: FeatureDict, complex_type: str, c_butype: str) -> FeatureDict:
    """
    Based on the config, apply filters and transformations to the data

    :param tensor: The feature dictionary for processing
    :param complex_type: The complex type, which is protein in the project
    :param c_butype: The number of types for biological units, it is 20 for protein residues
    """

    tensors = np_to_tensor_dict(np_example=tensors, features={ "num_btype", "butype", "between_segment_bu", "residue_index", "seq_length", "all_atom_positions", "all_atom_mask", "resolution", "is_distillation" })
    tensors["num_butype"] = torch.tensor(c_butype)
    nonensembled = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.cast_to_32bit_floats,
        data_transforms.squeeze_features,
        data_transforms.split_pos(ComplexType[complex_type]),
        data_transforms.make_seq_mask,
    ]

    nonensembled.append(data_transforms.make_dense_atom_masks(ComplexType[complex_type]))

    nonensembled.extend([ # supervised
        data_transforms.make_dense_atom_positions,
        data_transforms.all_atom_to_frames(),
        data_transforms.all_atom_to_torsion_angles(prefix=""),
        data_transforms.make_pseudo_beta(prefix=""),
        data_transforms.get_backbone_frames,
        data_transforms.get_chi_angles,
    ])

    for f in nonensembled:
        tensors = f(tensors)

    return tensors


def design_to_feature(proteins: Protein, complex_type: str, c_butype: str) -> FeatureDict:
    """
    Extract features from the protein

    :param protein: The protein that features come from
    :param complex_type: The complex type, which is protein in the project
    :param c_butype: The number of types for biological units, it is 20 for protein residues
    """

    features = []
    for protein in proteins:
        pdb_feature = make_pdb_features(protein, "whatever", is_distillation=False)
        feature = process_tensors_without_msa_template(pdb_feature, complex_type, c_butype)
        feature.pop("rna_pos")
        features.append(feature)
    def stack_fn(items: torch.Tensor) -> torch.Tensor:
        if len(items[0].size()) == 0:
            return torch.tensor(items)
        else:
            return torch.nn.utils.rnn.pad_sequence(items, batch_first=True, padding_value=0)

    features = dict_multimap(stack_fn, features)

    return features

