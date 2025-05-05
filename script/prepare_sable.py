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

import json
import os
import sys
import urllib.request

import lmdb
import pickle

from util import *


pdb_dir = os.path.join(DATA_ROOT, "pdb_20230501")
cath_chain_set_splits_url = "https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json" # the CATH protein design dataset split from work ``Generative Models for Graph-based Protein Design''
cath_chain_set_splits_path = os.path.split(cath_chain_set_splits_url)[-1]
ts50 = set(map(lambda x: x[ : 4].upper(), [
    "1eteA", "1v7mV", "1y1lA", "3pivA", "1or4A", "2i39A", "4gcnA", "1bvyF", "3on9A", "3vjzA",
    "3nbkA", "3l4rA", "3gwiA", "4dkcA", "3so6A", "3lqcA", "3gknA", "3nngA", "2j49A", "3fhkA",
    "2va0A", "3hklA", "2xr6A", "3ii2A", "2cayA", "3t5gB", "3ieyB", "3aqgA", "3q4oA", "2qdlA",
    "3ejfA", "3gfsA", "1ahsA", "2fvvA", "2a2lA", "3nzmA", "3e8mA", "3k7pA", "3ny7A", "2gu3A",
    "1pdoA", "1h4aX", "1dx5I", "1i8nA", "2cviA", "3a4rA", "1lpbA", "1mr1C", "2xcjA", "2xdgA"
])) # the TS50 list from work https://doi.org/10.1002/prot.24620
"""
casp_url_pattern = "https://predictioncenter.org/casp%d/targetlist.cgi?type=csv&view=regular"
casp_file_pattern = "sable_casp%d.list"
"""
casp_pdbid_set = set([
    "6POO", "6PX4", "6R17", "6S44", "6T1Z", "6UF2", "6UV6", "6VN1", "6VQP", "6VR4",
    "6XN8", "6XOD", "6Y4F", "6YGH", "6YJ1", "6ZYC", "7ABW", "7BGL", "7CN6", "7D2O",
    "7JTL", "7K7W", "7M5F", "7M6B", "7M7A", "7MHU", "7OC9", "7QG9", "7TH8", "7TXX",
    "7UM1", "7W6B", "8ONB", # CASP 14
    "7PBR", "7PZT", "7QIH", "7QVB", "7R1L", "7ROA", "7SQ4", "7TIL", "7UBZ", "7UUS",
    "7UWW", "7UX8", "7UYX", "7UZT", "7YR6", "7YR7", "7ZCX", "8A8C", "8AD2", "8BBT",
    "8C6Z", "8D5V", "8DYS", "8ECX", "8EM5", "8FEF", "8H2N", "8IFX", "8JVN", "8JVP",
    "8OK3", "8OKH", "8ON4", "8ORK", "8OUY", "8PBV", "8PKO", "8RJW", "8SMQ", "8SXA",
    "8TN8", "8UFN", "8XBP" # CASP 15
])
MIN_RESIDUES = 20


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_sable.py [output folder]")
        exit()

    eval_pdb_id_set = set()

    # obtain the protein design dataset test split
    urllib.request.urlretrieve(cath_chain_set_splits_url, cath_chain_set_splits_path)
    cath_test_split = json.loads(open(cath_chain_set_splits_path, "r").read())["test"]
    os.remove(cath_chain_set_splits_path)
    eval_pdb_id_set = set.union(ts50, set(map(lambda x: x[ : 4].upper(), cath_test_split)))

    # union with the model quality assessment dataset test split
    eval_pdb_id_set = set.union(eval_pdb_id_set, casp_pdbid_set)
    """
    for casp in [14, 15]: # constructure the test set
        casp_url = casp_url_pattern % (casp)
        casp_file = casp_file_pattern % (casp)
        urllib.request.urlretrieve(casp_url, casp_file)
        fin = open(casp_file, "r")
        casp_pdbid_set = set()
        for i, line in enumerate(fin):
            if i == 0 or not(line.strip()):
                continue
            description = line.split(";")[-1].strip()
            if "<em>" in description:
                description = description[ : description.index("<em>")]
            ids = description.split()
            if len(ids) > 0 and len(ids[-1]) == 4 and ids[-1].isalnum() and ids[-1].islower():
                casp_pdbid_set.add(ids[-1].upper())
        eval_pdb_id_set = set.union(eval_pdb_id_set, casp_pdbid_set)
        os.remove(casp_file)
    """

    eval_list = list(eval_pdb_id_set)

    os.makedirs(sys.argv[1], exist_ok=True)

    # prepare the validation data file and corresponding PDB ID set first
    eval_file_path = os.path.join(sys.argv[1], "eval.lmdb")
    if os.path.isfile(eval_file_path):
        os.remove(eval_file_path)
    env_out = lmdb.open(eval_file_path, subdir=False, lock=False, map_size=int(1e12))
    txn_out = env_out.begin(write=True)
    idx = 0
    for i, pdb_id in enumerate(eval_list):
        path = ""
        for ext in [".cif", ".pdb"]:
            path = os.path.join(pdb_dir, pdb_id + ext)
            if os.path.isfile(path):
                break
            path = ""
        if not(path):
            continue
        datapoint = extract_pdb(path, pdb_id)
        if datapoint and len(datapoint["residues"]) > MIN_RESIDUES:
            txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
            idx += 1
            print("\reval: %d/%d and %d kept" % (i + 1, len(eval_list), idx), end="")
            sys.stdout.flush()
    print("\reval: %d/%d and %d kept" % (i + 1, len(eval_list), idx))
    txn_out.commit()
    env_out.close()

    # prepare the train data file
    train_list = list(set(map(lambda x: x[ : 4], os.listdir(pdb_dir))))
    train_file_path = os.path.join(sys.argv[1], "train.lmdb")
    if os.path.isfile(train_file_path):
        os.remove(train_file_path)
    env_out = lmdb.open(train_file_path, subdir=False, lock=False, map_size=int(1e12))
    txn_out = env_out.begin(write=True)
    idx = 0
    for i, pdb_id in enumerate(train_list):
        if pdb_id in eval_pdb_id_set:
            continue
        path = ""
        for ext in [".cif", ".pdb"]:
            path = os.path.join(pdb_dir, pdb_id + ext)
            if os.path.isfile(path):
                break
            path = ""
        if not(path):
            continue
        datapoint = extract_pdb(path, pdb_id)
        if datapoint and len(datapoint["residues"]) > MIN_RESIDUES:
            txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
            idx += 1
            print("\rtrain: %d/%d and %d kept" % (i + 1, len(train_list), idx), end="")
            sys.stdout.flush()
    print("\rtrain: %d/%d and %d kept" % (i + 1, len(train_list), idx))
    txn_out.commit()
    env_out.close()

