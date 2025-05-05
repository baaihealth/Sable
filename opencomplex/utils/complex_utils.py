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

import enum
import torch


class CaseInsensitiveEnumMeta(enum.EnumMeta):
    def __getitem__(self, item):
        if isinstance(item, str):
            item = item.upper()
        return super().__getitem__(item)


class ComplexType(enum.Enum, metaclass=CaseInsensitiveEnumMeta):
    PROTEIN = 1
    RNA = 2
    MIX = 3


def correct_rna_butype(butype):
    if butype.numel() > 0 and torch.max(butype) > 7:
        return butype - 20
    return butype


def split_protein_rna_pos(bio_complex, complex_type=None):
    device = bio_complex["butype"].device

    protein_pos = []
    rna_pos = []
    if complex_type is None:
        complex_type = ComplexType.MIX if 'bio_id' in bio_complex else ComplexType.PROTEIN

    if complex_type == ComplexType.PROTEIN:
        protein_pos = torch.arange(0, bio_complex['butype'].shape[-1], device=device)
    elif complex_type == ComplexType.RNA:
        rna_pos = torch.arange(0, bio_complex['butype'].shape[-1], device=device)
    else:
        protein_pos = torch.where(bio_complex['bio_id'] == 0)[-1]
        rna_pos = torch.where(bio_complex['bio_id'] == 1)[-1]

    return protein_pos, rna_pos

