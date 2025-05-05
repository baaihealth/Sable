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

import gzip
import json
import os
import shutil
import sys
import urllib.request

import lmdb
import pickle

from util import *


all_structures_dir = os.path.join(DATA_ROOT, "all_structures") # download ZIP file from SAbDab through URL https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/ , and decompress as this directory
sabdab_summary_url = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/all/"
sabdab_summary_path = "sabdab_summary.tsv"
refinegnn_repo_url = "https://github.com/wengong-jin/RefineGNN.git"
heavy_chain_dir_pattern = "RefineGNN/data/sabdab/hcdr%d_cluster/"
abdockgen_repo_url = "https://github.com/wengong-jin/abdockgen.git" # two entries ("4lar" and "4kv5") missed as a result of the updating on sabdab summary
rabd_dir = "abdockgen/data/rabd/"
antigen_types = ["protein", "peptide"]
components = ["H", "L", "Ag"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_antibody_design.py [output folder]")
        exit()

    # obtain the SAbDab summary
    urllib.request.urlretrieve(sabdab_summary_url, sabdab_summary_path)
    column_indices = None
    chains = {}
    for line in open(sabdab_summary_path, "r"):
        tokens = line.split("\t")
        if not(column_indices):
            column_indices = [tokens.index("pdb"), tokens.index("Hchain"), tokens.index("Lchain"), tokens.index("antigen_chain"), tokens.index("antigen_type")]
        elif tokens[column_indices[1]] != "NA" and tokens[column_indices[-1]] in antigen_types:
            current_chains = [("" if tokens[i] == "NA" else tokens[i]) for i in column_indices[1 : -1]]
            if not(tokens[column_indices[0]] in chains) or (current_chains[1] and chains[tokens[column_indices[0]]][1] == ""):
                chains[tokens[column_indices[0]]] = current_chains
    for line in open(sabdab_summary_path, "r"):
        tokens = line.split("\t")
        if tokens[column_indices[0]] != "pdb" and tokens[column_indices[1]] != "NA" and not(tokens[column_indices[0]] in chains):
            current_chains = [("" if tokens[i] == "NA" else tokens[i]) for i in column_indices[1 : -1]]
            if current_chains[-1] == current_chains[0]:
                current_chains[-1] = ""
            chains[tokens[column_indices[0]]] = current_chains
    os.remove(sabdab_summary_path)

    os.makedirs(sys.argv[1], exist_ok=True)

    # process the three datasets for heavy chains
    os.system("git clone %s" % (refinegnn_repo_url))
    for hi in range(1, 4): # CDR H1, H2, and H3
        os.makedirs(os.path.join(sys.argv[1], "H%d" % (hi)), exist_ok=True)
        data_dir = heavy_chain_dir_pattern % (hi)
        files = os.listdir(data_dir)
        for file in files:
            split = file[ : file.index("_")]
            output_path = os.path.join(sys.argv[1], "H%d" % (hi), ("eval" if split == "val" else split) + ".lmdb")
            if os.path.isfile(output_path):
                os.remove(output_path)
            env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
            txn_out = env_out.begin(write=True)
            fin = gzip.open(os.path.join(data_dir, file), "rb") if file.endswith(".gz") else open(os.path.join(data_dir, file), "r")
            idx = 0
            for i, line in enumerate(fin):
                data = json.loads(line)
                pdb_id = data["pdb"]
                imgt_pdb_path = os.path.join(all_structures_dir, "imgt", pdb_id + ".pdb")
                if not(pdb_id in chains):
                    continue
                heavy_chain = chains[pdb_id][0]
                imgt_datapoint = extract_pdb(imgt_pdb_path, "%s_%s" % (pdb_id, heavy_chain), [heavy_chain], fv_region=True)
                if imgt_datapoint:
                    datapoint = {
                        "pdbid": imgt_datapoint["pdbid"],
                        "H_residues": imgt_datapoint["residues"],
                        "H_coordinates": imgt_datapoint["coordinates"],
                        "H_cdr": list(map(to_cdr, imgt_datapoint["sequence_number"])),
                    }
                    if str(hi) in datapoint["H_cdr"]:
                        datapoint["H_pdb"] = extract_full_pdb(imgt_pdb_path, imgt_datapoint["pdbid"], [heavy_chain])["H_pdb"]
                        txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                        idx += 1
                        print("\rH%d/%s: read %d lines and %d kept" % (hi, file, i + 1, idx), end="")
                        sys.stdout.flush()
            fin.close()
            print("\rH%d/%s: read %d lines and %d kept" % (hi, file, i + 1, idx))
            txn_out.commit()
            env_out.close()
    shutil.rmtree(heavy_chain_dir_pattern.split(os.sep)[0])

    # process the RAbD dataset
    os.system("git clone %s" % (abdockgen_repo_url))
    os.makedirs(os.path.join(sys.argv[1], "RAbD"), exist_ok=True)
    files = os.listdir(rabd_dir)
    for file in files:
        if not("jsonl" in file):
            continue
        split = file[ : file.index("_")]
        output_path = os.path.join(sys.argv[1], "RAbD", ("eval" if split == "val" else split) + ".lmdb")
        if os.path.isfile(output_path):
            os.remove(output_path)
        env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
        txn_out = env_out.begin(write=True)
        fin = gzip.open(os.path.join(rabd_dir, file), "rb") if file.endswith(".gz") else open(os.path.join(rabd_dir, file), "r")
        idx = 0
        for i, line in enumerate(fin):
            data = json.loads(line)
            pdb_id = data["pdb"]
            if not(pdb_id in chains):
                continue
            ab = chains[pdb_id]
            imgt_pdb_path = os.path.join(all_structures_dir, "imgt", pdb_id + ".pdb")
            datapoint = { "pdbid": "%s_%s" % (pdb_id, "_".join(filter(None, ab))) }
            for ci, c in enumerate(components):
                imgt_datapoint = extract_pdb(imgt_pdb_path, "%s_%s" % (pdb_id, ab[ci]), set(ab[ci].split(" | ")), fv_region=(c != components[-1])) if c else None
                if not(imgt_datapoint):
                    continue
                datapoint[c + "_residues"] = imgt_datapoint["residues"]
                datapoint[c + "_coordinates"] = imgt_datapoint["coordinates"]
                if c != components[-1]:
                    datapoint[c + "_cdr"] = list(map(to_cdr, imgt_datapoint["sequence_number"]))
                else:
                    datapoint[c + "_chainid"] = imgt_datapoint["chainid"]
            if sum(c + "_residues" in datapoint for c in components) == len(components):
                full_pdb_datapoint = extract_full_pdb(imgt_pdb_path, pdb_id, ab)
                if full_pdb_datapoint:
                    datapoint.update(full_pdb_datapoint)
                    txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                    idx += 1
                    print("\rRAbD/%s: read %d lines and %d kept" % (file, i + 1, idx), end="")
                    sys.stdout.flush()
        fin.close()
        print("\rRAbD/%s: read %d lines and %d kept" % (file, i + 1, idx))
        txn_out.commit()
        env_out.close()
    shutil.rmtree(rabd_dir.split(os.sep)[0])

