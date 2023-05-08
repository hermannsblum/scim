# MT Silvan Loew: Continual Learning for Novel Semantic Discovery in Unstructured Outdoor Environments
​
Based on SCIM: Simultaneous Clustering, Inference, and Mapping for Open-World Semantic Scene Understanding
[[paper](https://arxiv.org/abs/2206.10670)]


This repository provides our evluation data, pretrained models, and implementations of the tested methods.

# Model Checkpoints
To get started on experiments easily, we provide 10 DeepLabv3+ checkpoints trained on the OAISYS 16k RUGD dateset.

<table>
<tr><th>Model Nr</th><th>Checkpoint</th></tr>
<tr><td>1</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/Bm9N3GMqXzYqqbm">download</a></td></tr>
<tr><td>2</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/T3yNOEcXnXDmoVn">download</a></td></tr>
<tr><td>3</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/dY8Q7xbpkyFXNWw">download</a></td></tr>
<tr><td>4</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/RJcGXjDPXTQ7AIJ">download</a></td></tr>
<tr><td>5</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/T9sQkAVav7jEYyL">download</a></td></tr>
<tr><td>6</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/alu7UZvzxEaFomD">download</a></td></tr>
<tr><td>7</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/8NJbDfjJCpinAp6">download</a></td></tr>
<tr><td>8</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/OWhNgr9XijNaga8">download</a></td></tr>
<tr><td>9</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/vnKp1P3nZ4AY39C">download</a></td></tr>
<tr><td>10</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/3FggFu422AUg8rX">download</a></td></tr>
</table>

# Evaluation Data

Download the created Datasets here an untar them into `home/tensorflow_datasets/`
<table>
<tr><th>Name</th><th>Link</th></tr>
<tr><td>OAISYS 16k RUGD</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/3jcTNtm2rcnFzkB">download</a></td></tr>
<tr><td>OAISYS Trajectory Sand</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/Zbcnlk5MkJHyvaY">download</a></td></tr>
<tr><td>OAISYS Trajectory Grass-Field</td><td><a target="_blank" href="https://polybox.ethz.ch/index.php/s/AJepOzE0EqJBBy4">download</a></td></tr>
</table>

# Algorithm Implementations

The modular Algorithm is split up into several steps. Below we describe the workflow for the chosen methods and the step for each individual method.

<details>
  <summary>Proposed Algorithm</summary>

1. run inference
```bash
python3 deeplab/oaisys_ensemble.py with dataname=$DATASET device=cpu 
```

2. get the deeplab features and run parameter optimisation and clustering
```bash
python3 deeplab/oaisys_seg.py best_hdbscan with subset=$DATASET pretrained_model=$MODEL n_calls=100 cluster_selection_method=eom
```
3. create pseudolabels
```bash
deeplab/oaisys_pseudolabel.py with subset=$DATASET  outlier=$CLUSTER_NAME inlier=$INLIER_NAME uncert=$UNCERT_NAME  uncertainty_threshold=$UNCERT_THRESHOLD
```

4. adapt ensemble model
```bash
python3 deeplab/oaisys_adaptation_ensemble.py with subset=$DATASET pseudolabels=$PSEUDOLABEL_NAME pretrain_dataset=oaisys16k_rugd
```

5. run inference with adapted ensemble model
```bash
python3 deeplab/oaisys_adaptedinference_ensemble.py with subset=$DATASET pretrained_models=$MODELS_LIST pseudolabels=$PSEUDOLABEL_NAME training=$TRAINING_ID
```
</details>

<details>
  <summary>Single model</summary>
Replace the following steps with:

1. run inference
```bash
python3 deeplab/oaisys_inference.py with pretrained_model=$MODEL set=$DATASET
```
4. adapt  model
```bash
python3 deeplab/oaisys_adaptation.py with subset=$DATASET pseudolabels=$PSEUDOLABEL_NAME pretrain_dataset=oaisys16k_rugd
```
5. run inference with adapted model
```bash
python3 deeplab/oaisys_adaptedinference.py with subset=$DATASET pretrained_model=$MODEL pseudolabels=$PSEUDOLABEL_NAME training=$TRAINING_ID
```
  </details>

<details>
  <summary>Standardize max logit</summary>
Follow procedure of single model. Replace step 1 with:

1.1 calculate normaization values
```bash
python3 deeplab/oaisys_normalize_max_logit.py with pretrained_model=$MODEL 
```

1.2. run inference
```bash
python3 deeplab/oaisys_inference.py with pretrained_model=$MODEL set=$DATASET training_set=oaisys16k_rugd normalize_max_logits=True 
```
</details>

<details>
  <summary>GMM Post processing</summary>
Before step 1, run the following 2 steps:

0.1. fit gmm model
```bash
python3 deeplab/oaisys_gmm.py gmm_fit with pretrained_model=$MODEL subset=oaisys16k_rugd split=train num_classes=11 subsample=100
```

0.2. fit gmm model
```bash
python3 deeplab/oaisys_gmm.py gmm_predict with pretrained_model=$MODEL training_set=oaisys16k_rugd split=train num_classes=11 subsample=100 subset=$DATASET
```
</details>

<details>
  <summary>DML Net</summary>
We only evaluated our this OOD detection method for our dataset. Follow these steps for reproduction:

```bash
git clone https://github.com/loewsi/Open-World-Semantic-Segmentation.git
cd Open-World-Semantic-Segmentation
git checkout devel-silvan
cd anomaly
python3 train_embedding.py
python3 eval_ood_traditional.py --ood dissum

```
</details>

# Evaluation

There are a few jupyter notebooks provided for evaluation in 
`scim/evaluation`
# Installation

We offer a [dockerfile](https://github.com/hermannsblum/scim/blob/main/Dockerfile) that installs the whole code-base into a container. To install individual parts, see below:

## Clustering & Learning
This part is implemented in python. To install it, run:
```bash
git clone https://github.com/hermannsblum/scim.git
cd scim && python -m pip install -e .
git checkout silvan-devel 
```


## Data Structure

All intermediate outputs of different steps are stored to a folder. This folder needs to be set correctly in some places:

Add a file `semsegcluster/settings.py` with the following content:
```python
EXPERIMENT_STORAGE_FOLDER = '<folder for experimental logs>'
TMPDIR = '/tmp'
TMP_DIR =  '/tmp'
EXP_OUT = '<folder for outputs>'
```
