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

import _io
import os
from subprocess import Popen, PIPE
import sys
sys.path.append("..")
import tempfile
from typing import Any, Mapping, Sequence, Set, Tuple

import torch
from Bio.PDB import *
import numpy

from opencomplex.np import residue_constants


DATA_ROOT = "/share/project/chenxi/data/Sable/sable_source"
backbone_atoms = ["N", "CA", "C", "O"]
fv_max_sequence_number = 128


def extract_pdb(path: str, id: str, chain_filter: Set[str]=set(), fv_region: bool=False, verbose: bool=False) -> Mapping[str, Any]:
    """
    Extract basic informaiton (resudie, chain ID, and coordinate) of some chains in a protein from PDB

    :param path: The path for the PDB/CIF file, its extension is important for choosing parser
    :param id: The ID string used to distinguish proteins
    :param chain_filter: The filter indicating the set of chains to keep, empty set indicates keeping all chains
    :param fv_region: The indicator for Fv region on antibody, if so, residues with sequence number no more than 129 are kept and they are kept
    :param verbose: The indicator for whether to include sequence number and icode data
    """
    parser = PDBParser(QUIET=True) if not(path.endswith(".cif")) else MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(id, path)
    except:
        return None

    residues, chainids, coordinates, sequence_number, icode = [], [], [], [], []
    model = list(structure.get_models())[0]
    for chain in model:
        if chain_filter and not(chain.id in chain_filter):
            continue
        for residue in chain:
            if residue.id[0] != " " or not(residue.resname in residue_constants.restype_3to1): # similar to `is_aa` to make sure it is an amino acid
                continue
            if fv_region and residue.id[1] > fv_max_sequence_number:
                continue
            residues.append(residue_constants.restype_3to1[residue.resname])
            chainids.append(chain.id)
            mean_coordinates = sum(atom.coord for atom in residue) / len(residue) # is used in place where coordinates for backbone atoms are missing
            mean_coordinates = np.around(mean_coordinates, decimals=2)
            coordinates += [residue[atom_name].coord if atom_name in residue else mean_coordinates for atom_name in backbone_atoms]
            sequence_number.append(residue.id[1])
            icode.append(residue.id[2])
    ret = None
    if len(residues) > 0:
        ret = {
            "pdbid": id,
            "residues": residues,
            "chainid": chainids,
            "coordinates": numpy.expand_dims(numpy.stack(coordinates), axis=0),
        }
        if fv_region or verbose:
            ret["sequence_number"] = sequence_number
            ret["icode"] = icode
    return ret


def extract_full_pdb(path: str, id: str, chain_ids: Sequence[str]) -> Mapping[str, Mapping[str, numpy.ndarray]]:
    """
    Generate dictionary for full informaiton from PDB file, used by antibody design task

    :param path: The path for PDB file
    :param id: The ID string used to distinguish proteins
    :param chain_ids: The chain IDs for heavy chain, light chain, and antigen. chains in antigen are separated by "|"
    """
    ab_chains = {}
    for i, x in enumerate(chain_ids): # number chains for heavy chain/light chain/antigen as 0/1/2
        for c in x.split("|"):
            if c.strip() in ab_chains:
                ab_chains[c.strip()].append(i)
            else:
                ab_chains[c.strip()] = [i]
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(id, path)
    except:
        return None
    model = list(structure.get_models())[0]
    asym_id_set = {} # use chain ID and sequence number pair to distinguish residues
    asym_id, atom_coordinate, atom_mask, atom_temperature_factor, residue_chain_id, residue_id, residue_type_index = [[[] for j in range(3)] for i in range(7)] # prepare slots for heavy chain/light chain/antigen
    for chain in model:
        if not(chain.id in ab_chains): # the pdb data is filtered that only heavy/light/antigen chains are included
            continue
        chain_indices = ab_chains[chain.id]
        for residue in chain:
            if residue.id[0] != ' ' or not(residue.resname in residue_constants.restype_3to1): # similar to `is_aa` to make sure it is an amino acid
                continue
            if residue.id[1] > fv_max_sequence_number and not(2 in chain_indices): # for heavy/light chains, skip it
                continue
            coordinate = numpy.zeros((residue_constants.atom_type_num, 3, ))
            mask = numpy.zeros((residue_constants.atom_type_num, ))
            temperature_factor = numpy.zeros((residue_constants.atom_type_num, ))
            for atom in residue:
                if atom.name not in residue_constants.atom_types:
                    continue
                coordinate[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.
                temperature_factor[residue_constants.atom_order[atom.name]] = atom.bfactor
            for chain_index in chain_indices:
                if chain_index != 2 and residue.id[1] > fv_max_sequence_number:
                    continue
                atom_coordinate[chain_index].append(coordinate)
                atom_mask[chain_index].append(mask)
                atom_temperature_factor[chain_index].append(temperature_factor)
                residue_chain_id[chain_index].append(chain.id)
                residue_id[chain_index].append(residue.id[1])
                if not((chain.id, residue.id[2]) in asym_id_set):
                    asym_id_set[(chain.id, residue.id[2])] = len(asym_id_set) + 1
                asym_id[chain_index].append(asym_id_set[(chain.id, residue.id[2])])
                residue_shortname = residue_constants.restype_3to1[residue.resname]
                residue_type_index[chain_index].append(residue_constants.restype_order[residue_shortname])
    data = {}
    for i in range(len(chain_ids)):
        data[["H", "L", "Ag"][i] + "_pdb"] = {
            "asym_id": numpy.array(asym_id[i]),
            "atom_positions": numpy.array(atom_coordinate[i]),
            "atom_mask": numpy.array(atom_mask[i]),
            "b_factors": numpy.array(atom_temperature_factor[i]),
            "chain_index": numpy.array(residue_chain_id[i]),
            "residue_index": numpy.array(residue_id[i]),
            "aatype": numpy.array(residue_type_index[i]),
        }
    return data


def to_cdr(sequence_number: int) -> str:
    """
    The CDR region given sequence number using the rule for IMGT

    :param sequence_number: the sequence number information
    """

    if 27 <= sequence_number <= 38:
        return "1"
    elif 56 <= sequence_number <= 65:
        return "2"
    elif 105 <= sequence_number <= 117:
        return "3"
    else:
        return "0"


def write_pdb(serial_number: int, sequence_number: int, chain_id: str, residue: Residue.Residue, atom: Atom.Atom, pdb_out: _io.TextIOWrapper) -> None:
    """
    Write one atom line of PDB file

    :param serial_number: The re-numbered serial number to avoid wrong format in PDB for huge protein
    :param sequence_number: The sequence number comes from the model protein to align with the reference protein
    :param chain_id: The chain ID from the reference protein
    :param residue: The residue class which provides residue name and segment ID
    :param atom: The atom class which provides informations such as name, coordinates, occupancy, etc.
    :param pdb_out: The handler for PDB file to write to
    """
    pdb_out.write("ATOM  %5d %s%s%3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s  \n" % (serial_number,
        (" %-3s" if len(atom.name) < 4 and len(atom.element) == 1 and atom.name.startswith(atom.element) else "%-4s") % atom.name,
        atom.altloc, residue.resname, chain_id, sequence_number, atom.coord[0], atom.coord[1], atom.coord[2],
        atom.occupancy, atom.bfactor, residue.segid, atom.element))


def structure_information(tmscore_path: str, prediction_path: str, native_path: str, id: str) -> Mapping[str, Any]:
    """
    Calculate the GDT-TS, global LDDT and lDDT together, and pack the data

    :param tmscore_path: The path of executable file path for computing both TM and GDT-TS score
    :param prediction_path: The path of the PDB file for the prediction structure
    :param native_path: The path of the PDB file for the native structure
    :param id: The ID string used to distinguish proteins
    """
    gdt_ts_prefix = "GDT-TS-score= "
    global_lddt_prefix = "Global LDDT score: "
    lddt_start_line = "Chain\tResName\tResNum\tAsses.\tScore\t(Conserved/Total, over 4 thresholds)"

    # It is verified:
    #   all PDBs of `prediction_path` are single-chain one
    #   all sequence numbers of decoys8000k are consecutive
    #   all icodes are empty for these dataset, use sequence number directly
    #   all residue names are in the 20 residue types
    parser = PDBParser(QUIET=True) if not(prediction_path.endswith(".cif")) else MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(id, prediction_path)
    except:
        return None
    p_chain = sorted(list(list(list(structure.get_models())[0])[0]), key=lambda x: x.id[1])

    parser = PDBParser(QUIET=True) if not(native_path.endswith(".cif")) else MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(id, native_path)
    except:
        return None
    model = list(structure.get_models())[0]
    best = (0, " ")
    # This function should be very slow on CASP predictions for proteins with long and many chains
    #   Biopython
    #     There are "compound" information in the PDB header, but destroied (all chain IDs are lowercased and hard to distinguish) directly by Biopython
    #     Ideally, one only needs to check one chain in each entity. Take 6VN1 (for T1036s1) as an example, it reduces the number of chains to check from 9 to 3
    #     It is also possible to recover chain ID compound information easily from CASP, but may not be true for more common case, just ignore it
    #       Uppercase a whole compound group directly when they do not appear before
    #       Keep lowercased ones, otherwise
    #   lddt: its implementation very slow for long chains
    for chain in model: # pick the closed chain to the prediction with aligner
        n_chain = sorted(list(chain), key=lambda x: x.id[1])
        aligns = {} # alignments that sequence number differences are used as keys
        for p_residue in p_chain: # alignments with single sequence number difference
            for n_residue in n_chain:
                diff = p_residue.id[1] - n_residue.id[1]
                if diff in aligns or p_residue.resname != n_residue.resname:
                    continue
                pi, ni, align = 0, 0, []
                # each element of `align` is (pi, ni, length, accumulated length from left, accumulated length from right)
                while pi < len(p_chain):
                    while ni < len(n_chain) and n_chain[ni].id[1] < p_chain[pi].id[1] - diff:
                        ni += 1
                    if ni == len(n_chain):
                        break
                    l = 0
                    while pi + l < len(p_chain) and ni + l < len(n_chain) and diff == p_chain[pi + l].id[1] - n_chain[ni + l].id[1] and p_chain[pi + l].resname == n_chain[ni + l].resname:
                        l += 1
                    if l:
                        align.append([pi, ni, l])
                        pi += l
                        ni += l
                    else:
                        pi += 1
                for i in range(len(align)): # accumulated length from left
                    align[i].append(align[i - 1][-1] + align[i][2] if i else align[i][2])
                for i in range(len(align) - 1, -1, -1): # accumulated length from right
                    align[i].append(align[i + 1][-1] + align[i][2] if i + 1 < len(align) else align[i][2])
                aligns[diff] = align
                if align[0][-1] > best[0]:
                    best = (align[0][-1], chain.id, n_chain, diff, len(p_chain))
        if not(aligns):
            continue
        for right_diff, r in aligns.items(): # concatenate 2 alignments with different difference
            for left_diff, l in aligns.items():
                if left_diff == right_diff or l[0][-1] + r[0][-1] <= best[0]:
                    continue
                ri = 0
                for li in range(0, len(l)):
                    while ri < len(r) and r[ri][0] < l[li][0] + l[li][2]:
                        ri += 1
                    if ri >= len(r):
                        break
                    if r[ri][0] == l[li][0] + l[li][2] and r[ri][1] >= l[li][1] + l[li][2] and l[li][-2] + r[ri][-1] > best[0]:
                        best = (l[li][-2] + r[ri][-1], chain.id, n_chain, left_diff, r[ri][0], right_diff, len(p_chain))

    tempfile_name_prefix = next(tempfile._get_candidate_names()) # crop the prediction chain
    p_tempfile_name = tempfile_name_prefix + "_m.pdb" # generate new PDB file by filtering out conflict residues
    n_tempfile_name = tempfile_name_prefix + "_r.pdb"
    p_out = open(p_tempfile_name, "w")
    n_out = open(n_tempfile_name, "w")
    kept_residues = {}
    chain_id, n_chain = best[1], best[2]
    pi = ni = 0
    sequence_number = p_serial_number = n_serial_number = 0
    for i in range(3, len(best), 2):
        diff, tail = best[i], best[i + 1]
        while pi < tail:
            if ni == len(n_chain): # no native residues left
                sequence_number += 1
                for atom in p_chain[pi]:
                    p_serial_number += 1
                    write_pdb(p_serial_number, sequence_number, chain_id, p_chain[pi], atom, p_out)
                pi += 1
            elif diff == p_chain[pi].id[1] - n_chain[ni].id[1]: # residue from both chain
                if p_chain[pi].resname == n_chain[ni].resname: # residues without confliction
                    sequence_number += 1
                    for atom in p_chain[pi]:
                        p_serial_number += 1
                        write_pdb(p_serial_number, sequence_number, chain_id, p_chain[pi], atom, p_out)
                    kept_residues[sequence_number] = n_chain[ni]
                    for atom in n_chain[ni]:
                        n_serial_number += 1
                        write_pdb(n_serial_number, sequence_number, chain_id, n_chain[ni], atom, n_out)
                pi += 1
                ni += 1
            elif p_chain[pi].id[1] < diff + n_chain[ni].id[1]: # residue from prediction chain
                sequence_number += 1
                for atom in p_chain[pi]:
                    p_serial_number += 1
                    write_pdb(p_serial_number, sequence_number, chain_id, p_chain[pi], atom, p_out)
                pi += 1
            else: # residue from native chain
                sequence_number += 1
                kept_residues[sequence_number] = n_chain[ni]
                for atom in n_chain[ni]:
                    n_serial_number += 1
                    write_pdb(n_serial_number, sequence_number, chain_id, n_chain[ni], atom, n_out)
                ni += 1
    while ni < len(n_chain): # residue from native chain
        sequence_number += 1
        kept_residues[sequence_number] = n_chain[ni]
        for atom in n_chain[ni]:
            n_serial_number += 1
            write_pdb(n_serial_number, sequence_number, chain_id, n_chain[ni], atom, n_out)
        ni += 1
    p_out.close()
    n_out.close()

    gdt_ts = 0.0
    process = Popen([os.path.abspath(tmscore_path), p_tempfile_name, n_tempfile_name], stdout=PIPE, stderr=PIPE, text=True)
    stdout, _ = process.communicate()
    lines = stdout.split("\n")
    for line in lines:
        if line.startswith(gdt_ts_prefix):
            gdt_ts = float(line[len(gdt_ts_prefix) : line.index(" ", len(gdt_ts_prefix))])

    global_lddt = 0.0
    residues, chainids, coordinates, lddt = [], [], [], []
    lddt_start = False
    process = Popen(["lddt", "-t", p_tempfile_name, n_tempfile_name], stdout=PIPE, stderr=PIPE, text=True)
    stdout, _ = process.communicate()
    lines = stdout.split("\n")
    for line in lines:
        line = line.strip(" ")
        if not(line):
            continue
        if lddt_start:
            tokens = line.split("\t")
            if tokens[4] == "-": # the lddt column
                continue
            residue = kept_residues.get(int(tokens[2]), None) # the sequence number field
            if residue:
                residues.append(residue_constants.restype_3to1[residue.resname])
                chainids.append(chain_id)
                mean_coordinates = sum(atom.coord for atom in residue) / len(residue) # is used in place where coordinates for backbone atoms are missing
                mean_coordinates = np.around(mean_coordinates, decimals=2)
                coordinates += [residue[atom_name].coord if atom_name in residue else mean_coordinates for atom_name in backbone_atoms]
                lddt.append(float(tokens[4]))
        elif line.startswith(global_lddt_prefix):
            global_lddt = float(line[len(global_lddt_prefix) : ])
        elif line == lddt_start_line:
            lddt_start = True

    os.remove(p_tempfile_name)
    os.remove(n_tempfile_name)

    return {
        "pdbid": "%s_%s" % (id, chain_id),
        "residues": residues,
        "chainid": chainids,
        "coordinates": numpy.expand_dims(numpy.stack(coordinates), axis=0),
        "gdt_ts": gdt_ts,
        "global_lddt": global_lddt,
        "lddt": numpy.expand_dims(numpy.array(lddt), axis=0),
    }

