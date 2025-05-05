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

"""Constants used in RNAFold."""

from typing import Mapping

import numpy as np
import torch

# Format: The list for each NT type contains delta, gamma, beta, alpha1, alpha1, tm, chi in this order. 
chi_angles_atoms = {
    'A': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N9", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N9", "C4"],
    ],
    'U': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N1", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N1", "C2"],
    ],
    'G': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N9", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N9", "C4"],
    ],
    'C': [
        ["C5'", "C4'", "C3'", "O3'"],
        ["C3'", "C4'", "C5'", "O5'"],
        ["C4'", "C5'", "O5'", "P"],
        ["C5'", "O5'", "P", "OP1"],
        ["C5'", "O5'", "P", "OP2"],
        ["N1", "C1'", "C2'", "O2'"],
        ["C2'", "C1'", "N1", "C2"],
    ]
}

# If chi angles given in fixed-length array, this matrix determines how to mask
# them for each AA type. The order is as per restype_order.
# In the order of delta, gamma, beta, alpha1, alpha1, tm, chi
chi_angles_mask = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # A
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # U
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # G
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # C
]

# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    "A": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4", "O2'", "C3'", "C2'", "OP1", "OP2", "N1", "N3", "N6", "N7", "C2", "C5", "C6", "C8"],
    "U": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2", "O2'", "C3'", "C2'", "OP1", "OP2", "N3", "C4", "C5", "C6", "O2", "O4"],
    "G": ["C5'", "C4'", "O4'", "N9", "C1'", "O5'", "P", "O3'", "C4", "O2'", "C3'", "C2'", "OP1", "OP2", "N1", "N2", "N3", "N7", "C2", "C5", "C6", "C8", "O6"],
    "C": ["C5'", "C4'", "O4'", "N1", "C1'", "O5'", "P", "O3'", "C2", "O2'", "C3'", "C2'", "OP1", "OP2", "N3", "N4", "C4", "C5", "C6", "O2"],
}

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    "C1'", 
    "C2'", 
    "C3'", 
    "C4'", 
    "C5'", 
    "O5'", 
    "O4'", 
    "O3'", 
    "O2'", 
    "P", 
    "OP1",
    "OP2",
    "N1", 
    "N2", 
    "N3", 
    "N4", 
    "N6", 
    "N7", 
    "N9", 
    "C2", 
    "C4", 
    "C5", 
    "C6", 
    "C8", 
    "O2", 
    "O4", 
    "O6",
]

atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# A compact atom encoding with 23 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom23_names = {
    "A": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N9", "O2'", "C4", "N1", "N3", "N6", "N7", "C2", "C5", "C6", "C8", ""],
    "U": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N1", "O2'", "C2", "N3", "C4", "C5", "C6", "O2", "O4", "", "", ""],
    "G": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N9", "O2'", "C4", "N1", "N2", "N3", "N7", "C2", "C5", "C6", "C8", "O6"],
    "C": ["C3'", "C4'", "O4'", "C2'", "C1'", "C5'", "O3'", "O5'", "P", "OP1", "OP2", "N1", "O2'", "C2", "N3", "N4", "C4", "C5", "C6", "O2", "", "", ""],
}

# This is the standard residue order when coding AA type as a number.
# Reproduce it by taking 3-letter AA codes and sorting them alphabetically.
restypes = [
    "A",
    "G",
    "C",
    "U"
]


def sequence_to_onehot(
    sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool = False
) -> np.ndarray:
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An RNA sequence.
      mapping: A dictionary mapping nucleotides to integers.
      map_unknown_to_x: If True, any nucleotide that is not in the mapping will be
        mapped to the unknown nucleotide 'X'. If the mapping doesn't contain
        nucleotide 'X', an error will be thrown. If False, any nucleotide not in
        the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_nts) with one-hot encoding of
      the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_nts - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1

    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            "The mapping must have values from 0 to num_unique_aas-1 "
            "without any gaps. Got: %s" % sorted(mapping.values())
        )

    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)

    for nt_index, nt_type in enumerate(sequence):
        if map_unknown_to_x:
            if nt_type.isalpha() and nt_type.isupper():
                nt_id = mapping.get(nt_type, mapping["X"])
            else:
                raise ValueError(
                    f"Invalid character in the sequence: {nt_type}"
                )
        else:
            nt_id = mapping[nt_type]
        one_hot_arr[nt_index, nt_id] = 1

    return one_hot_arr


def get_restype_atom_mapping(device="cpu"):
    restype_atom23_to_atom27 = []
    restype_atom27_to_atom23 = []
    restype_atom23_mask = []

    for rt in restypes:
        atom_names = restype_name_to_atom23_names[rt]
        restype_atom23_to_atom27.append(
            [(atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx23 = {name: i for i, name in enumerate(atom_names)}
        restype_atom27_to_atom23.append(
            [
                (atom_name_to_idx23[name] if name in atom_name_to_idx23 else 0)
                for name in atom_types
            ]
        )

        restype_atom23_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    restype_atom23_to_atom27 = torch.tensor(
        restype_atom23_to_atom27,
        dtype=torch.int32,
        device=device,
    )
    restype_atom27_to_atom23 = torch.tensor(
        restype_atom27_to_atom23,
        dtype=torch.int32,
        device=device,
    )
    restype_atom23_mask = torch.tensor(
        restype_atom23_mask,
        dtype=torch.float32,
        device=device,
    )

    # create the corresponding mask
    restype_atom27_mask = torch.zeros(
        [4, 27], dtype=torch.float32, device=device
    )
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_letter
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            restype_atom27_mask[restype, atom_type] = 1

    return restype_atom23_to_atom27, restype_atom27_to_atom23, restype_atom23_mask, restype_atom27_mask

