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

"""Protein data type."""
import dataclasses
from typing import Sequence, Optional

from opencomplex.np import residue_constants
import numpy as np

from opencomplex.utils.complex_utils import ComplexType


# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # The numerical ID for insertion code, starting from 1
    asym_id: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB 
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    def get_chain_sequence(self, idx):
        return residue_constants.restype_to_str_sequence(self.aatype[self.chain_index==idx])

    def get_chain_type(self, _):
        return ComplexType.PROTEIN

    @property
    def butype(self):
        return self.aatype


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
  chain_end = 'TER'
  return (f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} '
          f'{chain_name:>1}{residue_index:>4}')


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ['X']
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    asym_id = prot.asym_id.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError('Invalid aatypes.')

    # Construct a mapping from chain integer indices to chain ID strings.
    chain_ids = {}
    for i in np.unique(chain_index):  # np.unique gives sorted output.
        if i >= PDB_MAX_CHAINS:
            raise ValueError(
                f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]

    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]
    # Add all atom sites.
    for i in range(aatype.shape[0]):
        # Close the previous chain if in a multichain PDB.
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(
                atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]],
                residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1  # Atom index increases at the TER symbol.

        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]):

            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ' ' if asym_id[i] == 1 else PDB_CHAIN_IDS[asym_id[i] - 2] # borrow table from chains
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ''
            # PDB is a columnar format, every space matters here!
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
                f'{residue_index[i]:>4}{insertion_code:>1}   '
                f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                f'{occupancy:>6.2f}{b_factor:>6.2f}          '
                f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1

    # Close the final chain.
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                                chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')

    # Pad all lines to 80 characters.
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'  # Add terminating newline.

