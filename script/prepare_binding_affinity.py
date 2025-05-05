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

import csv
import os
import sys
sys.path.append("..")
import random
import shutil
import sys

import lmdb
import pickle

from opencomplex.np import residue_constants
from util import *


default_pdb_dir = os.path.join(DATA_ROOT, "SKEMPI2_PDBs") # download and decompressed from https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz
geoppi_repo_url = "https://github.com/Liuxg16/GeoPPI.git"
data_dir = "GeoPPI/data/benchmarkDatasets/"
seed = 42
corrections = { "1.00E+50": "1E50", "1.00E+96": "1E96" }
chain_modifier = { # it is for M1101
    **dict.fromkeys([("1DQJ", "H"), ("1MLC", "H"), ("1N8Z", "H"), ("1VFB", "H")], "B"),
    **dict.fromkeys([("1DQJ", "L"), ("1MLC", "L"), ("1N8Z", "L"), ("1VFB", "L")], "A"),
    **dict.fromkeys([("3BN9", "A")], "B"),
    **dict.fromkeys([("3BN9", "H"), ("2NYY", "H"), ("1YY9", "H"), ("2NZ9", "H")], "D"),
    **dict.fromkeys([("3BN9", "L"), ("2NYY", "L"), ("1YY9", "L"), ("2NZ9", "L")], "C"),
}
pdb_dir_updater = {
    **dict.fromkeys(["2I9B", "1KBH"], os.path.join(DATA_ROOT, "pdb_20230501")),
}
sequence_number_shift = {
    **dict.fromkeys([("S1131.csv", "2I9B", "E")], 8),
    **dict.fromkeys([("M1707.csv", "1KBH", "A"), ("S4169.csv", "1KBH", "A")], -893),
    **dict.fromkeys([("M1707.csv", "1KBH", "B"), ("S4169.csv", "1KBH", "B")], -1074),
}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_binding_affinity.py [output folder]")
        exit()

    # process the three datasets for heavy chains
    os.system("git clone %s" % (geoppi_repo_url))

    os.makedirs(sys.argv[1], exist_ok=True)

    random.seed(seed)
    datasets = os.listdir(data_dir)
    s8338_datapoints = []
    for dataset in datasets:
        if len(dataset) != 9 or not(dataset.endswith(".csv")):
            continue
        os.makedirs(os.path.join(sys.argv[1], dataset[ : -4]), exist_ok=True)
        fin = open(os.path.join(data_dir, dataset), newline="", encoding="ISO-8859-1")
        rdr = csv.reader(fin)
        datalines = []
        mutation_chians = {} # chain IDs appear in the mutation
        for line in rdr:
            if rdr.line_num == 1:
                i_mutation = len(line) - 1 - next((i for i, v in enumerate(line[ :: -1]) if v.lower().startswith("mutation")), -1)
                i_ddg = next((i for i, v in enumerate(line) if v.lower().startswith("ddg")), -1)
                i_label = next((i for i, v in enumerate(line) if v == "Label"), -1)
            else: # columns are line number, PDB ID, partner, mutation, and ddg
                pdb_id = corrections.get(line[0], line[0])
                if not(pdb_id in mutation_chians):
                    mutation_chians[pdb_id] = set()
                reverse = (i_label >= 0 and line[i_label] == "reverse") # the reverse case, which happen in M1707 set
                mutations = dict()
                for mutation in line[i_mutation].split(","):
                    tokens = mutation.strip().split(":")
                    if all(x.isalnum() for x in tokens):
                        if len(tokens) == 1:
                            chain_id, w_residue, m_residue, seq = tokens[0][1], tokens[0][0], tokens[0][-1], tokens[0][2 : -1]
                        elif len(tokens) == 2:
                            chain_id, w_residue, m_residue, seq = tokens[0], tokens[1][0], tokens[1][-1], tokens[1][1 : -1]
                        mutation_chians[pdb_id].add(chain_id)
                        if seq[-1].isdigit():
                            sequence_number, icode = int(seq), ' '
                        else:
                            sequence_number, icode = int(seq[ : -1]), seq[-1].upper()
                        if w_residue != m_residue and w_residue in residue_constants.restype_1to3 and m_residue in residue_constants.restype_1to3:
                            mutations[(chain_modifier.get((pdb_id[-4 : ], chain_id), chain_id), sequence_number + sequence_number_shift.get((dataset, pdb_id, chain_id), 0), icode, w_residue)] = m_residue
                if len(mutations) > 0:
                    datalines.append((len(datalines), pdb_id, line[1], mutations, float(line[i_ddg]), reverse))
        random.shuffle(datalines)
        partitions = [("eval.lmdb", int(len(datalines) * 0.1)), ("train.lmdb", int(len(datalines) * 0.9)), ("test.lmdb", len(datalines))]
        pi = 0
        for i, dataline in enumerate(datalines):
            if i == 0 or (pi > 0 and i == partitions[pi - 1][1]):
                output_path = os.path.join(sys.argv[1], dataset[ : -4], partitions[pi][0])
                if os.path.isfile(output_path):
                    os.remove(output_path)
                env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
                txn_out = env_out.begin(write=True)
                idx = 0
            path = ""
            for ext in [".pdb", ".cif"]:
                path = os.path.join(os.path.join(DATA_ROOT, "pdb_20230501") if dataset == "M1101.csv" else pdb_dir_updater.get(dataline[1], default_pdb_dir), dataline[1][-4 : ] + ext)
                if os.path.isfile(path):
                    break
                path = ""
            if path:
                chain_filter = set(list(map(lambda x: chain_modifier.get((dataline[1][-4 : ], x), x), filter(lambda y: y != "_", [*(dataline[2])]))))
                wild_datapoint = extract_pdb(path, dataline[1][-4 : ], chain_filter=(chain_filter if dataset.startswith("S") or dataset == "M1101.csv" else set()), verbose=True)
                if wild_datapoint:
                    datapoint = {
                        "binding_free_energy_change": dataline[4],
                        "chainid": wild_datapoint["chainid"],
                        "mutant_coordinates": wild_datapoint["coordinates"],
                        "partners": dataline[2],
                        "pdbid": "%s_%d" % (dataline[1], dataline[0]),
                        "wild_coordinates": wild_datapoint["coordinates"],
                    }
                    if dataline[5]: # keep binding free energy change, and switch wild/mutatnt for the reverse case
                        datapoint.update({
                            "wild_residues": [dataline[3][k] if k in dataline[3] else k[3] for k in zip(wild_datapoint["chainid"], wild_datapoint["sequence_number"], wild_datapoint["icode"], wild_datapoint["residues"])],
                            "mutant_residues": wild_datapoint["residues"],
                        })
                    else:
                        datapoint.update({
                            "wild_residues": wild_datapoint["residues"],
                            "mutant_residues": [dataline[3][k] if k in dataline[3] else k[3] for k in zip(wild_datapoint["chainid"], wild_datapoint["sequence_number"], wild_datapoint["icode"], wild_datapoint["residues"])],
                        })
                    if sum(1 if x != y else 0 for x, y in zip(datapoint["mutant_residues"], datapoint["wild_residues"])) == len(dataline[3]):
                        txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                        idx += 1
                        print("\r%s/%s: read %d/%d lines and %d kept" % (dataset, partitions[pi][0], i + 1, len(datalines), idx), end="")
                        sys.stdout.flush()
                        if dataset[ : -4] == "S4169": # the S8338 dataset is constructed from the S4169 set by switch wild/mutatnt and take the opposite number of binding free energy change
                            s8338_datapoints.append(pickle.dumps(datapoint))
                            datapoint = {
                                "binding_free_energy_change": -datapoint["binding_free_energy_change"],
                                "chainid": datapoint["chainid"],
                                "mutant_residues": datapoint["wild_residues"],
                                "mutant_coordinates": datapoint["wild_coordinates"],
                                "partners": datapoint["partners"][ :: -1],
                                "pdbid": datapoint["pdbid"] + "_mutant",
                                "wild_residues": datapoint["mutant_residues"],
                                "wild_coordinates": datapoint["mutant_coordinates"],
                            }
                            s8338_datapoints.append(pickle.dumps(datapoint))
                    else:
                        pass
            if i == partitions[pi][1] - 1:
                print("\r%s/%s: read %d/%d lines and %d kept" % (dataset, partitions[pi][0], i + 1, len(datalines), idx))
                txn_out.commit()
                env_out.close()
                pi += 1

    if len(s8338_datapoints) > 0:
        os.makedirs(os.path.join(sys.argv[1], "S8338"), exist_ok=True)
        random.shuffle(s8338_datapoints)
        partitions = [("eval.lmdb", int(len(s8338_datapoints) * 0.1)), ("train.lmdb", int(len(s8338_datapoints) * 0.9)), ("test.lmdb", len(s8338_datapoints))]
        pi = 0
        for i, datapoint in enumerate(s8338_datapoints):
            if i == 0 or (pi > 0 and i == partitions[pi - 1][1]):
                output_path = os.path.join(sys.argv[1], "S8338", partitions[pi][0])
                if os.path.isfile(output_path):
                    os.remove(output_path)
                env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
                txn_out = env_out.begin(write=True)
                idx = 0
            txn_out.put(str(idx).encode("ascii"), datapoint)
            idx += 1
            print("\rS8338/%s: %d/%d and %d kept" % (partitions[pi][0], i + 1, len(s8338_datapoints), idx), end="")
            sys.stdout.flush()
            if i == partitions[pi][1] - 1:
                print("\rS8338/%s: %d/%d and %d kept" % (partitions[pi][0], i + 1, len(s8338_datapoints), idx))
                txn_out.commit()
                env_out.close()
                pi += 1

    shutil.rmtree(data_dir.split(os.sep)[0])

