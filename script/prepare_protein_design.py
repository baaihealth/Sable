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
import json
import os
import sys
import urllib.request

import lmdb
import pickle

from util import *


pdb_dirs = [
    os.path.join(DATA_ROOT, "pdb_20230501"),
    os.path.join(DATA_ROOT, "pdb_missed"),
]
cath_chain_set_splits_url = "https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json" # the CATH protein design dataset split from work ``Generative Models for Graph-based Protein Design''
cath_chain_set_splits_path = os.path.split(cath_chain_set_splits_url)[-1]
ts50_remove_url = "https://raw.githubusercontent.com/drorlab/gvp-pytorch/main/data/ts50remove.txt" # the remove list from work ``Learning from Protein Structure with Geometric Vector Perceptrons''
ts50_remove_path = os.path.split(ts50_remove_url)[-1]
ts50 = list(map(lambda x: "%s.%s" % (x[ : 4], x[4]), [
    "1eteA", "1v7mV", "1y1lA", "3pivA", "1or4A", "2i39A", "4gcnA", "1bvyF", "3on9A", "3vjzA",
    "3nbkA", "3l4rA", "3gwiA", "4dkcA", "3so6A", "3lqcA", "3gknA", "3nngA", "2j49A", "3fhkA",
    "2va0A", "3hklA", "2xr6A", "3ii2A", "2cayA", "3t5gB", "3ieyB", "3aqgA", "3q4oA", "2qdlA",
    "3ejfA", "3gfsA", "1ahsA", "2fvvA", "2a2lA", "3nzmA", "3e8mA", "3k7pA", "3ny7A", "2gu3A",
    "1pdoA", "1h4aX", "1dx5I", "1i8nA", "2cviA", "3a4rA", "1lpbA", "1mr1C", "2xcjA", "2xdgA"
])) # the TS50 list from work https://doi.org/10.1002/prot.24620

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_protein_design.py [output root]")
        exit()

    # obtain the dataset splits
    urllib.request.urlretrieve(cath_chain_set_splits_url, cath_chain_set_splits_path)
    splits = json.loads(open(cath_chain_set_splits_path, "r").read())
    os.remove(cath_chain_set_splits_path)

    # obtain the remove set
    urllib.request.urlretrieve(ts50_remove_url, ts50_remove_path)
    ts50_remove = list(filter(None, map(lambda x: x.strip(), open(ts50_remove_path, "r").readlines())))
    os.remove(ts50_remove_path)

    # construct two datasets
    os.makedirs(sys.argv[1], exist_ok=True)
    for (dataset, remove_list), (split, output_filename) in product([("CATH", []), ("TS50", ts50_remove)], [("train", "train.lmdb"), ("validation", "eval.lmdb"), ("test", "test.lmdb")]):
        os.makedirs(os.path.join(sys.argv[1], dataset), exist_ok=True)
        output_path = os.path.join(sys.argv[1], dataset, output_filename)
        if os.path.isfile(output_path):
            os.remove(output_path)
        env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
        txn_out = env_out.begin(write=True)
        chain_list = list(filter(lambda x: not(x in remove_list), splits[split]))
        if dataset == "TS50" and split == "test":
            chain_list = ts50
        idx = 0
        for i, chain in enumerate(chain_list):
            path = ""
            for pdb_dir, ext in product(pdb_dirs, [".pdb", ".cif"]):
                path = os.path.join(pdb_dir, chain[ : 4].upper() + ext)
                if os.path.isfile(path):
                    break
                path = ""
            if not(path):
                continue
            datapoint = extract_pdb(path, chain, set([*(chain[4 : ])]))
            if datapoint:
                txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                idx += 1
            print("\r%s/%s: %d/%d with %d kept" % (dataset, split, i + 1, len(chain_list), idx), end="")
            sys.stdout.flush()
        print("\r%s/%s: %d/%d with %d kept" % (dataset, split, i + 1, len(chain_list), idx))
        txn_out.commit()
        env_out.close()

