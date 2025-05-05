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

from itertools import product
import os
import shutil
import sys

import lmdb
import pickle

from util import *


pdb_dirs = [os.path.join(DATA_ROOT, "pdb_20230501"), os.path.join(DATA_ROOT, "pdb_missed")]
repo_url = "https://github.com/phermosilla/IEConv_proteins.git"
label_dir = "IEConv_proteins/Datasets/data/ProtFunct"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_enzyme-catalyzed_reaction_classification.py [output folder]")
        exit()

    os.system("git clone %s" % (repo_url))
    os.makedirs(sys.argv[1], exist_ok=True)
    class_map = dict(list(map(lambda x: x.strip().split(","), open(os.path.join(label_dir, "chain_functions.txt"), "r").readlines())))

    label_files = os.listdir(label_dir)
    for label_file, output_filename in [("testing.txt", "test"), ("training.txt", "train"), ("validation.txt", "eval")]:
        output_path = os.path.join(sys.argv[1], output_filename + ".lmdb")
        if os.path.isfile(output_path):
            os.remove(output_path)
        env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
        txn_out = env_out.begin(write=True)
        protein_list = list(map(lambda x: x.strip(), open(os.path.join(label_dir, label_file), "r").readlines()))
        idx = 0
        for i, protein in enumerate(protein_list):
            tokens = protein.split(".")
            path = ""
            for pdb_dir, ext in product(pdb_dirs, [".pdb", ".cif"]):
                path = os.path.join(pdb_dir, tokens[0].upper() + ext)
                if os.path.isfile(path):
                    break
                path = ""
            if not(path):
                continue
            datapoint = extract_pdb(path, protein, set(tokens[1 : ]))
            if datapoint:
                datapoint["class"] = class_map[protein]
                txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                idx += 1
            print("\r%s: %d/%d and %d kept" % (output_filename, i + 1, len(protein_list), idx), end="")
            sys.stdout.flush()
        print("\r%s: %d/%d and %d kept" % (output_filename, i + 1, len(protein_list), idx))
        txn_out.commit()
        env_out.close()
    shutil.rmtree(label_dir.split(os.sep)[0])

