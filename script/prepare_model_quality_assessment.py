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
import random
import sys
import urllib.request

import lmdb
import pickle

from util import *


pdb_dir = os.path.join(DATA_ROOT, "pdb_20230501")
decoys8000k_dir = os.path.join(DATA_ROOT, "decoys8000k") # download ZIP file from https://files.ipd.uw.edu/pub/DeepAccNet/decoys8000k.zip mentioned by DeepAccNet repository, and decompress as this directory
train_eval_list_path = "model_quality_assessment.tsv" # sampling takes long, use the original list without making changes for evaluation consistency
seed = 42
native_pdb_filename = "native.pdb"
decoys_sample_count = 5
tmscore_source_url = "https://zhanglab.dcmb.med.umich.edu/TM-score/TMscore.cpp"
tmscore_source_path = "TMscore.cpp"
tmscore_path = "TMscore"
casp_url_pattern = "https://predictioncenter.org/casp%d/targetlist.cgi?type=csv&view=regular"
casp_file_pattern = "mqa_casp%d.list"
casp_predictions_dir_pattern = os.path.join(DATA_ROOT, "CASP_predictions") + "/CASP%d/predictions/regular/"
test_set = set([
    "T1024", "T1025", "T1026", "T1027", "T1028", "T1029", "T1030", "T1031", "T1033", "T1035", # CASP14
    "T1036s1", "T1037", "T1039", "T1040", "T1041", "T1042", "T1043", "T1045s1", "T1045s2", "T1046s1",
    "T1046s2", "T1047s1", "T1047s2", "T1049", "T1051", "T1053", "T1055", "T1056", "T1057", "T1058",
    "T1059", "T1060s2", "T1060s3", "T1064", "T1065s1", "T1065s2", "T1072s1", "T1072s2", "T1074", "T1076",
    "T1082", "T1089", "T1090", "T1091", "T1092", "T1093", "T1094", "T1095", "T1096", "T1099",
    "T1104", "T1106s1", "T1106s2", "T1114s1", "T1114s2", "T1114s3", "T1119", "T1120", "T1121", "T1123", # CASP15
    "T1124", "T1129s2", "T1133", "T1134s1", "T1134s2", "T1137s1", "T1137s2", "T1137s3", "T1137s4", "T1137s5",
    "T1137s6", "T1137s7", "T1137s8", "T1137s9", "T1152", "T1159", "T1170", "T1187",
])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_model_quality_assessment.py [output root]")
        exit()

    if not(os.path.isfile(tmscore_path)):
        urllib.request.urlretrieve(tmscore_source_url, tmscore_source_path)
        os.system("g++ -static -O3 -ffast-math -lm -o %s %s" % (tmscore_path, tmscore_source_path))
        os.remove(tmscore_source_path)

    os.makedirs(sys.argv[1], exist_ok=True)

    random.seed(seed)
    datalines = list(filter(None, map(lambda x: x.strip(), open(train_eval_list_path, "r").readlines())))
    partitions = [("eval.lmdb", 741), ("train.lmdb", len(datalines))]
    pi = 0
    for i, dataline in enumerate(datalines):
        if i == 0 or (pi > 0 and i == partitions[pi - 1][1]):
            output_path = os.path.join(sys.argv[1], partitions[pi][0])
            if os.path.isfile(output_path):
                os.remove(output_path)
            env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
            txn_out = env_out.begin(write=True)
            idx = 0
        decoys_root = os.path.join(decoys8000k_dir, dataline)
        decoys = os.listdir(decoys_root)
        random.shuffle(decoys)
        decoys = decoys[ : decoys_sample_count]
        for decoy in decoys:
            datapoint = structure_information(tmscore_path, os.path.join(decoys_root, decoy), os.path.join(decoys_root, native_pdb_filename), "%s_%s" % (dataline, decoy[ : -4]))
            if datapoint:
                txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                idx += 1
            if pi == 0:
                print("\r%s: %d/%d targets and %d kept" % (partitions[pi][0], i + 1, partitions[pi][1], idx), end="")
            else:
                print("\r%s: %d/%d targets and %d kept" % (partitions[pi][0], i + 1 - partitions[pi - 1][1], partitions[pi][1] - partitions[pi - 1][1], idx), end="")
            sys.stdout.flush()
        if i == partitions[pi][1] - 1:
            if pi == 0:
                print("\r%s: %d/%d targets and %d kept" % (partitions[pi][0], i + 1, partitions[pi][1], idx))
            else:
                print("\r%s: %d/%d targets and %d kept" % (partitions[pi][0], i + 1 - partitions[pi - 1][1], partitions[pi][1] - partitions[pi - 1][1], idx))
            txn_out.commit()
            env_out.close()
            pi += 1

    for casp in [14, 15]: # constructure the test set
        os.makedirs(os.path.join(sys.argv[1], "CASP%d_test" % (casp)), exist_ok=True)
        output_path = os.path.join(sys.argv[1], "CASP%d_test" % (casp), "test.lmdb")
        if os.path.isfile(output_path):
            os.remove(output_path)
        env_out = lmdb.open(output_path, subdir=False, lock=False, map_size=int(1e12))
        txn_out = env_out.begin(write=True)
        casp_predictions_dir = casp_predictions_dir_pattern % (casp)
        targetlist = os.listdir(casp_predictions_dir)
        casp_url = casp_url_pattern % (casp)
        casp_file = casp_file_pattern % (casp)
        urllib.request.urlretrieve(casp_url, casp_file)
        fin = open(casp_file, "r")
        idx = target_count = prediction_count = 0
        for i, line in enumerate(fin):
            if i == 0 or not(line.strip()):
                continue
            columns = line.split(";")
            if not(columns[0] in test_set) or not(columns[0] in targetlist):
                continue
            description = columns[-1].strip()
            if "<em>" in description:
                description = description[ : description.index("<em>")]
            ids = description.split()
            if not(len(ids) > 0 and len(ids[-1]) == 4 and ids[-1].isalnum() and ids[-1].islower()):
                continue
            path = ""
            for ext in [".pdb", ".cif"]:
                path = os.path.join(pdb_dir, ids[-1].upper() + ext)
                if os.path.isfile(path):
                    break
                path = ""
            if not(path):
                continue
            target_count += 1
            target_dir = os.path.join(casp_predictions_dir, columns[0])
            predictions = os.listdir(target_dir)
            for j, prediction in enumerate(predictions):
                datapoint = structure_information(tmscore_path, os.path.join(target_dir, prediction), path, "%s_%s_%s" % (columns[0], prediction, ids[-1].upper()))
                if datapoint:
                    txn_out.put(str(idx).encode("ascii"), pickle.dumps(datapoint))
                    idx += 1
                print("\rCASP%d: %d targets %d predictions done, and %d kept" % (casp, target_count, prediction_count + j + 1, idx), end="")
                sys.stdout.flush()
            prediction_count += len(predictions)
        print("\rCASP%d: %d targets %d predictions done, and %d kept" % (casp, target_count, prediction_count, idx))
        txn_out.commit()
        env_out.close()
        os.remove(casp_file)

