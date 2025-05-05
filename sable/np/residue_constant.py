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

from opencomplex.np import residue_constants as rc


restype_num = rc.restype_num
atom_type_num = rc.atom_type_num
special_token_count = 5
dictionary_size = restype_num + special_token_count
PAD, BOS, EOS, UNK = range(0, special_token_count - 1)
restype_begin = UNK + 1
restype_end = restype_begin + restype_num
restype_weights = torch.tensor([0] * restype_begin + [1] * restype_num, dtype=torch.float)
MASK = dictionary_size - 1


def residue_id_lookup(residue: str) -> int:
    """
    Map the residue symbol to a numerical ID, coming from the order in OpenComplex

    :param residue: the single-character residue symbol to encode
    """

    return rc.restype_order[residue] + restype_begin

def residue_type_explain(residue_id: int) -> str:
    """
    Map the numerical ID of residue to its sing-character symbol, apply "X" when it is not found

    :param residue_id: the numerical ID for the residue
    """

    return rc.restypes[residue_id - restype_begin] if restype_begin <= residue_id < restype_end else "X"

