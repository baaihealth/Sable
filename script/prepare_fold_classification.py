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

import os
import shutil
import sys

import lmdb
import pickle

from util import *


pdb_dir = os.path.join(DATA_ROOT, "pdbstyle-1.75") # the decompressed PDB files directory from https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-1.75-[1-4].tgz
repo_url = "https://github.com/multicom-toolbox/DeepSF.git"
fold_dir = "DeepSF/datasets/D2_Three_levels_dataset"
rename_map = { "Traindata.list": "train.lmdb", "test_dataset.list_family": "test_family", "test_dataset.list_fold": "test_fold", "test_dataset.list_superfamily": "test_superfamily", "validation.list": "eval.lmdb" }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_fold_classification.py [output folder]")
        exit()

    os.system("git clone %s" % (repo_url))
    os.makedirs(sys.argv[1], exist_ok=True)
    tags = set()
    all_protein_folds = {}
    for list_file, output_name in rename_map.items():
        protein_folds = []
        fin = open(os.path.join(fold_dir, list_file), "r")
        for line in fin:
            tokens = line.strip().split()
            if tokens:
                protein_folds.append((tokens[0], tokens[-1]))
                tags.add(tokens[-1])
        fin.close()
        all_protein_folds[output_name] = protein_folds
    folds = list(map(lambda x: (x[0], int(x[1])), map(lambda y: y.split("."), tags)))
    folds.sort()
    folds = dict(map(lambda x: ("%s.%d" % (x[1][0], x[1][1]), str(x[0])), enumerate(folds)))

    for output_name, protein_folds in all_protein_folds.items():
        if "test" in output_name:
            os.makedirs(os.path.join(sys.argv[1], output_name), exist_ok=True)
            output_path = os.path.join(sys.argv[1], output_name, "test.lmdb")
        else:
            output_path = os.path.join(sys.argv[1], output_name)
        if os.path.isfile(output_path):
            os.remove(output_path)
        env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
        txn_out = env_out.begin(write=True)
        idx = 0
        for i, (protein, fold) in enumerate(protein_folds):
            datapoint = extract_pdb(os.path.join(pdb_dir, protein[2 : 4], (protein if protein[-1] == "_" or not("_" in protein) else protein.replace("_", ".")) + ".ent"), protein)
            if datapoint:
                datapoint["class"] = folds[fold]
                txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                idx += 1
            print("\r%s: %d/%d and %d kept" % (output_name, i + 1, len(protein_folds), idx), end="")
            sys.stdout.flush()
        print("\r%s: %d/%d and %d kept" % (output_name, i + 1, len(protein_folds), idx))
        txn_out.commit()
        env_out.close()
    shutil.rmtree(fold_dir.split(os.sep)[0])

