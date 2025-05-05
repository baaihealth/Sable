# Before Running

* Take the symbol that the total number of residues in a protein is $N_{\rm res}$
* Only coordinates of backbone atoms are used in this work, which backbone atoms come in the order of $N$, $C_\alpha$, $C$, and $O$

# Data Preparation

All early example scripts used for preparing training/validation/test data locate in **`script`**[^1] subfolder (you can customize them for different training requirements), with similar command line
```bash
python prepare_[task name].py [output folder]
```

## Pre-downloaded Data

There are some data that should be prepared manually since they are huge in size and time-consuming to download. It assumes that they are downloaded and decompressed beforehand in directory pointed by _`DATA_ROOT`_ variable in **`script/util.py`**

An example directory of the _`DATA_ROOT`_ (set as **`/data/sable`**) looks like

<details>
<summary>The example structure of the DATA_ROOT directory</summary>
<pre>/data/sable/
├── CASP_predictions
├── SKEMPI2_PDBs
├── all_structures
├── decoys8000k
├── pdb_20230501
├── pdb_missed
└── pdbstyle-1.75</pre>
</details>

### **`pdb_20230501`**

This the directory for PDBx/mmCIF snapshot resources at timestamp 2023-05-01. The full list could be obtained by [querying RCSB PDB](https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22and%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_accession_info.initial_release_date%22%2C%22operator%22%3A%22less_or_equal%22%2C%22negation%22%3Afalse%2C%22value%22%3A%222023-05-01%22%7D%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%5D%2C%22label%22%3A%22text%22%7D%5D%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options%22%3A%7B%22paginate%22%3A%7B%22start%22%3A0%2C%22rows%22%3A25%7D%2C%22results_content_type%22%3A%5B%22experimental%22%5D%2C%22sort%22%3A%5B%7B%22sort_by%22%3A%22score%22%2C%22direction%22%3A%22desc%22%7D%5D%2C%22scoring_strategy%22%3A%22combined%22%7D%2C%22request_info%22%3A%7B%22query_id%22%3A%223dce2a5d84b11b28e3dae77fdf5c7e72%22%7D%7D), and downloaded into this directory. Their [recommanded way of downloading](https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script) could be applied.

### **`decoys8000k`**

The decoys data used to constructure the training/validation set for model quality assessment task. It is mentioned in the [README file](https://github.com/hiranumn/DeepAccNet/blob/main/README.md#resources) of [DeepAccNet repository](https://github.com/hiranumn/DeepAccNet). It is a huge package and could be [downloaded](https://files.ipd.uw.edu/pub/DeepAccNet/decoys8000k.zip) and decompressed directly

Note: this set is de-duplicated with a threshold of 40% and is done by [PISCES server](https://dunbrack.fccc.edu/pisces/). The de-duplicated list is provided directly as [model_quality_assessment.tsv](./script/model_quality_assessment.tsv), since one has to wait the result from the reply mail

### **`CASP_predictions`**

The predictions for [CASP14](https://predictioncenter.org/casp14/)/[CASP15](https://predictioncenter.org/casp15/) as the test set for model quality assessment task. They could be downloaded from the [download area](https://www.predictioncenter.org/download_area)

The path pattern looks like `https://www.predictioncenter.org/download_area/[CASP ID]/predictions/regular/[Target ID].tar.gz`, which the _`CASP ID`_ could be _`CASP14`_ or _`CASP15`_. An example for target [T1024](https://predictioncenter.org/casp14/target.cgi?id=8&view=all) from CASP14 looks like [**`T1024.tar.gz`**](https://www.predictioncenter.org/download_area/CASP14/predictions/regular/T1024.tar.gz)

The **`CASP_predictions`** directory is organized the same structure as the download area, and targets are decompressed in seperated subdirectories. The structure looks like

<details>
<summary>The example structure of the CASP_predictions directory</summary>
<pre>CASP_predictions/
├── CASP14
│   └── predictions
│       └── regular
│           ├── T1024
│           │   ├── T1024TS004_1
│           │   ├── T1024TS004_2
│           │   ├── T1024TS004_3
│           │   ├── ...
│           ├── T1027
│           │   ├── ...
│           ├── ...
└── CASP15
    └── predictions
        └── regular
            ├── T1104
            │   ├── T1104TS003_1
            │   ├── T1104TS003_2
            │   ├── T1104TS003_3
            │   ├── ...
            ├── T1105
            │   ├── ...
            ├── ...</pre>
</details>

### **`SKEMPI2_PDBs`**

This data package contains all the main PDB resources for binding affinity task. It could be [downloaded](https://life.bsc.es/pid/skempi2/database/download/SKEMPI2_PDBs.tgz) and decompressed directly

### **`all_structures`**

The full PDB resources from the structural antibody database [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab) used by the antibody design task. It could be [downloaded](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/archive/all/) and decompressed directly

### **`pdbstyle-1.75`**

The PDB resources used by the fold classifications. It could be constructured by downloading and decompressing four files in the URL pattern `https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-1.75-[1-4].tgz`

### **`pdb_missed`**

The PDB resources missing at the time of 2023-05-01 used by enzyme-catalyzed reaction classification and protein design tasks. They could be downloaded from the RCSB directly, and there are 28 of them

<details>
<summary>The PDB items in pdb_missed directory</summary>
<pre>1BVS
2I6L
2KEA
2MNT
2MUO
3AKE
3EHD
3FSP
3KWU
3OHM
3R5Q
3WWY
4ELN
4GBO
4OTW
4ROR
4WTO
5A0J
5D95
5HBG
5JN1
5OBL
5URR
5X8O
5XWQ
6FED
6GT1
6IHZ</pre>
</details>

# Sable (Pre-train)

The command for starting a **Sable** pre-training
```bash
python go_sable.py
```

<details>
<summary>The example structure of the data folder</summary>
<pre>example_data/sable
├── eval.lmdb
└── train.lmdb</pre>
</details>

<details>
<summary>The structure detail for one datapoint</summary>
<table style="text-align: center">
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Value Type</th>
            <th>Value Size</th>
            <th>Value Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>residues</td>
            <td>Symbols for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['A', 'G', ..., 'D']</td>
        </tr>
        <tr>
            <td>chainid</td>
            <td>Chain ID for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['A', 'A', ..., 'A']</td>
        </tr>
        <tr>
            <td>coordinates</td>
            <td>Coordinates for residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[9.43, 27.207, 36.902], [9.56, 25.91, 36.169], ..., [31.867, 43.18, 31.964]], dtype=float32)]</td>
        </tr>
    </tbody>
</table>
</details>

Here's an example of output folder (in _`paths.log_dir`_) structure for a pre-training run. The 7th checkpoint (`best_006.ckpt`) is the best one according to validation metrics, the 6th (`005.ckpt`), the 7th (`006.ckpt`), and the 10th (`009.ckpt`) are the top 3, in the **`checkpoints`** subfolder

<details>
<summary>The example structure of the output folder for a pre-training run</summary>
<pre>log_dir/sable/runs/CIF_2024-03-17_12-12-26
├── checkpoints
│   ├── 005.ckpt
│   ├── 006.ckpt
│   ├── 009.ckpt
│   ├── best_006.ckpt
│   └── last.ckpt
├── config_tree.log
├── sable.log
├── tags.log
└── wandb
    ├── debug-internal.log -> offline-run-20240317_121228-rivkb9p0/logs/debug-internal.log
    ├── debug.log -> offline-run-20240317_121228-rivkb9p0/logs/debug.log
    ├── latest-run -> offline-run-20240317_121228-rivkb9p0
    └── offline-run-20240317_121228-rivkb9p0
        ├── files
        │   ├── conda-environment.yaml
        │   ├── wandb-metadata.json
        │   └── wandb-summary.json
        ├── logs
        │   ├── debug-internal.log
        │   └── debug.log
        ├── run-rivkb9p0.wandb
        └── tmp
            └── code
</pre>
</details>

# Enzyme-catalyzed Reaction Classification

The command for starting a training from scratch
```bash
python go_sable.py run=enzyme-catalyzed_reaction_classification
```

The command for fine-tuning given pre-trained checkpoint (the _`ckpt_path`_ argument)
```bash
python go_sable.py run=enzyme-catalyzed_reaction_classification +fine-tune=True trainer.max_epochs=20 ckpt_path=/path/to/checkpoint
```

The command for testing a model, here _`ckpt_path`_ is set as the checkpoint after training this task from scratch or fine-tuning instead of the pre-trained one
```bash
python go_sable.py run=enzyme-catalyzed_reaction_classification train=False ckpt_path=/path/to/checkpoint
```

<details>
<summary>The example structure of the data folder</summary>
<pre>example_data/enzyme-catalyzed_reaction_classification
├── eval.lmdb
├── test.lmdb
└── train.lmdb</pre>
</details>

<details>
<summary>The structure detail for one datapoint</summary>
<table style="text-align: center">
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Value Type</th>
            <th>Value Size</th>
            <th>Value Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>residues</td>
            <td>Symbols for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['G', 'P', ..., 'L']</td>
        </tr>
        <tr>
            <td>coordinates</td>
            <td>Coordinates for residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[9.317, -16.265, -7.107], [9.376, -16.753, -8.521], ..., [-20.989, -44.107, -30.148]], dtype=float32)]</td>
        </tr>
        <tr>
            <td>class</td>
            <td>The class label</td>
            <td>str</td>
            <td></td>
            <td>'357'</td>
        </tr>
    </tbody>
</table>
</details>

# Fold Classification

The command for starting a training from scratch
```bash
python go_sable.py run=fold_classification
```

The command for fine-tuning given pre-trained checkpoint (the _`ckpt_path`_ argument)
```bash
python go_sable.py run=fold_classification +fine-tune=True trainer.max_epochs=20 ckpt_path=/path/to/checkpoint
```

The command for testing a model, here _`ckpt_path`_ is set as the checkpoint after training this task from scratch or fine-tuning instead of the pre-trained one
```bash
python go_sable.py run=fold_classification train=False ckpt_path=/path/to/checkpoint
```

<details>
<summary>The example structure of the data folder</summary>
<pre>example_data/fold_classification
├── eval.lmdb
├── test_family
│   └── test.lmdb
├── test_fold
│   └── test.lmdb
├── test_superfamily
│   └── test.lmdb
└── train.lmdb</pre>
</details>

<details>
<summary>The structure detail for one datapoint</summary>
<table style="text-align: center">
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Value Type</th>
            <th>Value Size</th>
            <th>Value Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>residues</td>
            <td>Symbols for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['V', 'E', ..., 'H']</td>
        </tr>
        <tr>
            <td>coordinates</td>
            <td>Coordinates for residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[31.448, 18.78, 70.403], [31.591, 20.238, 70.697], ..., [9.186, 9.338, 71.461]], dtype=float32)]</td>
        </tr>
        <tr>
            <td>class</td>
            <td>The class label</td>
            <td>str</td>
            <td></td>
            <td>'0'</td>
        </tr>
    </tbody>
</table>
</details>

# Antibody Design

The command for starting a training from scratch (use _`dataset=RAbD`_ for example, the default value is _`dataset=H1`_)
```bash
python go_sable.py run=antibody_design dataset=RAbD
```

The command for fine-tuning given pre-trained checkpoint (the _`ckpt_path`_ argument)
```bash
python go_sable.py run=antibody_design dataset=RAbD +fine-tune=True trainer.max_epochs=20 ckpt_path=/path/to/checkpoint
```

The command for testing a model, here _`ckpt_path`_ is set as the checkpoint after training this task from scratch or fine-tuning instead of the pre-trained one
```bash
python go_sable.py run=antibody_design dataset=RAbD train=False ckpt_path=/path/to/checkpoint
```

<details>
<summary>The example structure of the data folder</summary>
<pre>example_data/antibody_design
├── H1
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
├── H2
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
├── H3
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
└── RAbD
    ├── eval.lmdb
    ├── test.lmdb
    └── train.lmdb</pre>
</details>

According to the description of complementarity-determining region (CDR) from [Wikipedia](https://en.wikipedia.org/wiki/Complementarity-determining_region),
> There are three CDRs (CDR1, CDR2 and CDR3), arranged non-consecutively, on the amino acid sequence of a variable domain of an antigen receptor. Since the antigen receptors are typically composed of two variable domains (on two different polypeptide chains, heavy and light chain), there are six CDRs for each antigen receptor that can collectively come into contact with the antigen.

we involve the notation (the `{component}` below) for components that "H" stands for heavy chain, "L" stands for the light chain, and "Ag" for antigen. What's more, CDRs on heavy or light chain are denoted as "1", "2", and "3", respectively

<details>
<summary>The structure detail for one datapoint</summary>
<table style="text-align: center">
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Value Type</th>
            <th>Value Size</th>
            <th>Value Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>{component}_residues</td>
            <td>Symbols for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['E', 'V', ..., 'S']</td>
        </tr>
        <tr>
            <td>{component}_coordinates</td>
            <td>Coordinates for residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[52.732, 19.493, 29.69], [52.243, 20.705, 30.391], ..., [16.756, 13.757, 48.357]], dtype=float32)]</td>
        </tr>
        <tr>
            <td>{component}_cdr</td>
            <td>The CDR labels</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['0', ..., '0', '1', ..., '1', '0', ..., '0', '2', ..., '2', '0', ..., '0', '3', ..., '3', '0', ..., '0']</td>
        </tr>
        <tr>
            <td>{component}_pdb</td>
            <td>Informaiton from PDB</td>
            <td>Mapping[str, numpy.ndarray]</td>
            <td></td>
            <td></td>
        </tr>
    </tbody>
</table>
</details>

The information in `{component}_pdb` dictionary comes from parsing results of [PDB](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) file
* the `pack_pdb` function in script [script/pack_pdb.py](./script/pack_pdb.py) could be used to generate it, given both the path of PDB file `pdb_file_path` and chain IDs `chain_ids` for different components
* one example of `chain_ids` parameter looks like "E_H_K | D", in which "_" separates components and "|" separates chain IDs. In this example, the "E" chain is the heavy chain, the "H" chain is the light chain, and "K" and "D" are the antigen

# Protein Design

The command for starting a training from scratch (use _`dataset=TS50`_ for example, the default value is _`dataset=CATH`_)
```bash
python go_sable.py run=protein_design dataset=TS50
```

The command for fine-tuning given pre-trained checkpoint (the _`ckpt_path`_ argument)
```bash
python go_sable.py run=protein_design dataset=TS50 +fine-tune=True trainer.max_epochs=20 ckpt_path=/path/to/checkpoint
```

The command for testing a model, here _`ckpt_path`_ is set as the checkpoint after training this task from scratch or fine-tuning instead of the pre-trained one
```bash
python go_sable.py run=protein_design dataset=TS50 train=False ckpt_path=/path/to/checkpoint
```

<details>
<summary>The example structure of the data folder</summary>
<pre>example_data/protein_design
├── CATH
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
└── TS50
    ├── eval.lmdb
    ├── test.lmdb
    └── train.lmdb</pre>
</details>

<details>
<summary>The structure detail for one datapoint</summary>
<table style="text-align: center">
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Value Type</th>
            <th>Value Size</th>
            <th>Value Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>residues</td>
            <td>Symbols for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['S', 'A', ..., 'Q']</td>
        </tr>
        <tr>
            <td>chainid</td>
            <td>Chain ID for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['A', 'A', ..., 'A']</td>
        </tr>
        <tr>
            <td>coordinates</td>
            <td>Coordinates for residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[123.898, -10.152,  122.808], [124.058, -9.42, 124.102], ..., [67.534,  8.474,  64.442]], dtype=float32)]</td>
        </tr>
    </tbody>
</table>
</details>

# Binding Affinity

The command for starting a training from scratch (use _`dataset=S1131`_ for example, the default value is _`dataset=M1101`_)
```bash
python go_sable.py run=binding_affinity dataset=S1131
```

The command for fine-tuning given pre-trained checkpoint (the _`ckpt_path`_ argument)
```bash
python go_sable.py run=binding_affinity dataset=S1131 +fine-tune=True trainer.max_epochs=20 ckpt_path=/path/to/checkpoint
```

The command for testing a model, here _`ckpt_path`_ is set as the checkpoint after training this task from scratch or fine-tuning instead of the pre-trained one
```bash
python go_sable.py run=binding_affinity dataset=S1131 train=False ckpt_path=/path/to/checkpoint
```

<details>
<summary>The example structure of the data folder</summary>
<pre>example_data/binding_affinity
├── M1101
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
├── M1707
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
├── S1131
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
├── S4169
│   ├── eval.lmdb
│   ├── test.lmdb
│   └── train.lmdb
└── S8338
    ├── eval.lmdb
    ├── test.lmdb
    └── train.lmdb</pre>
</details>

<details>
<summary>The structure detail for one datapoint</summary>
<table style="text-align: center">
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Value Type</th>
            <th>Value Size</th>
            <th>Value Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>wild_residues</td>
            <td>Symbols for wild-type residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['V', 'W', ..., 'C']</td>
        </tr>
        <tr>
            <td>wild_coordinates</td>
            <td>Coordinates for wild-type residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[-10.005, 61.502, -88.814], [-10.272, 60.859, -87.532], ..., [-7.949, -26.391, -59.023]], dtype=float32)]</td>
        </tr>
        <tr>
            <td>mutant_residues</td>
            <td>Symbols for mutant residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['V', 'W', ..., 'C']</td>
        </tr>
        <tr>
            <td>mutant_coordinates</td>
            <td>Coordinates for mutant residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[-10.005, 61.502, -88.814], [-10.272, 60.859, -87.532], ..., [-7.949, -26.391, -59.023]], dtype=float32)]</td>
        </tr>
        <tr>
            <td>binding_free_energy_change</td>
            <td>The experimentally determined changes in free energy </td>
            <td>float</td>
            <td></td>
            <td>0.14</td>
        </tr>
    </tbody>
</table>
</details>

# Model Quality Assessment

The command for starting a training from scratch
```bash
python go_sable.py run=model_quality_assessment
```

The command for fine-tuning given pre-trained checkpoint (the _`ckpt_path`_ argument)
```bash
python go_sable.py run=model_quality_assessment +fine-tune=True trainer.max_epochs=20 ckpt_path=/path/to/checkpoint
```

The command for testing a model, here _`ckpt_path`_ is set as the checkpoint after training this task from scratch or fine-tuning instead of the pre-trained one
```bash
python go_sable.py run=model_quality_assessment train=False ckpt_path=/path/to/checkpoint
```

<details>
<summary>The example structure of the data folder</summary>
<pre>example_data/model_quality_assessment
├── CASP14_test
│   └── test.lmdb
├── CASP15_test
│   └── test.lmdb
├── eval.lmdb
└── train.lmdb</pre>
</details>

<details>
<summary>The structure detail for one datapoint</summary>
<table style="text-align: center">
    <thead>
        <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Value Type</th>
            <th>Value Size</th>
            <th>Value Example</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>residues</td>
            <td>Symbols for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['S', 'V', ..., 'S']</td>
        </tr>
        <tr>
            <td>chainid</td>
            <td>Chain ID for residues</td>
            <td>Sequence[str]</td>
            <td>The length is N<sub>res</sub></td>
            <td>['A', 'A', ..., 'A']</td>
        </tr>
        <tr>
            <td>coordinates</td>
            <td>Coordinates for residues</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (4 * N<sub>res</sub>, 3)</td>
            <td>[array([[11.499, 37.244, 7.315], [11.757, 35.983, 6.629], ..., [32.767, -2.418, 48.14]], dtype=float32)]</td>
        </tr>
        <tr>
            <td>lddt</td>
            <td>The scores of local distance test</td>
            <td>Sequence[numpy.ndarray]<br>dtype=float32</td>
            <td>The numpy.ndarray's shape is (N<sub>res</sub>)</td>
            <td>[array([0.29, 0.4782, ..., 0.6261], dtype=float32)]</td>
        </tr>
        <tr>
            <td>gdt_ts</td>
            <td>The score of global distance test</td>
            <td>float</td>
            <td></td>
            <td>0.5301</td>
        </tr>
    </tbody>
</table>
</details>

[^1]: The command and file names are ```in code block```, argument and variables names are _```italic in code block```_, and paths are **```bold in code block```**

