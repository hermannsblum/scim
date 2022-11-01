# SCIM: Simultaneous Clustering, Inference, and Mapping for Open-World Semantic Scene Understanding

[[paper](https://arxiv.org/abs/2206.10670)]


This repository provides our evluation data, pretrained models, and implementations of the tested methods.

# Model Checkpoints

To get started on experiments easily, we provide DeepLabv3+ checkpoints trained on first COCO and then ScanNet, but excluding different classes:

<table>
<tr><th>Outlier Class</th><th>ScanNet Validation</th><th>Checkpoint</th></tr>
<tr><td>television</td><td>43% mIoU</td><td><a target="_blank" href="https://zenodo.org/record/6840795/files/deeplab_no_tv.pth?download=1">download</a></td></tr>
<tr><td>books + bookshelf</td><td>42% mIoU</td><td><a target="_blank" href="https://zenodo.org/record/6840795/files/deeplab_no_books.pth?download=1">download</a></td></tr>
<tr><td>towel</td><td>41% mIoU</td><td><a target="_blank" href="https://zenodo.org/record/6840795/files/deeplab_no_towel.pth?download=1">download</a></td></tr>
</table>

You can also load the models directly through torchhub:

```python
import torch
no_tv = torch.hub.load('hermannsblum/scim:main', 'dv3res101_no_tv')
no_book = torch.hub.load('hermannsblum/scim:main', 'dv3res101_no_book')
no_towel = torch.hub.load('hermannsblum/scim:main', 'dv3res101_no_towel')
```


# Evaluation Data

To automatically download and preprocess data, we use TFDS with a pytorch wrapper:

```python
import tensorflow_datasets as tfds
import semsegcluster.data.scannet
from semsegcluster.data.tfds_to_torch import TFDataIterableDataset

data = tfds.load('scan_net/scene0354_00', split='validation')
torchdata = TFDataIterableDataset(data)
```

# Method Implementations

Method implementations are split up into several steps for added flexibility. Below we describe the workflows for each method.

<details>
  <summary>Nakajima</summary>

1. run inference
```bash
python deeplab/scannet_inference.py with subset=$SCENE and pretrained_model=$MODEL
```
2. run mapping (for flexibility, we run semantic mapping and uncertainty mapping separately)
```bash
roslaunch panoptic_mapping_utils scannnet_mapping.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
roslaunch panoptic_mapping_utils scannnet_uncertmap.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
```
3. render the maps
```bash
panoptic_mapping_utils scannnet_predrender.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
panoptic_mapping_utils scannnet_voxelidrender.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
panoptic_mapping_utils scannnet_uncertrender.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
```
4. get the geometric features (we run 3DSmoothNet in a singularity container)
```bash
singularity run --nv --bind $OUTPUTS/$SCENE/$MODEL/point_cloud_0.ply:/pc.ply --bind $OUTPUTS/$SCENE/$MODEL/smoothnet:/output --bind $SMOOTHNET_DATA/evaluate:/3DSmoothNet/data/evaluate --bind $SMOOTHNET_DATA/logs:/3DSmoothNet/logs --bind $SMOOTHNET_DATA/preprocessed:/preprocessed smoothnet.simg
roslaunch panoptic_mapping_utils scannnet_geofeatures.launch scene:=$SCENE model:=$MODEL
```
5. run parameter optimisation and clustering
```bash
python3 deeplab/scannet_nakajima.py best_mcl_nakajima  with subset=$SCENE pretrained_model=$MODEL n_calls=100 shard=20
```
  </details>

<details>
  <summary>Uhlemeyer</summary>

1. run inference
```bash
python deeplab/scannet_inference.py with subset=$SCENE and pretrained_model=$MODEL
```

2. run meta-segmentation and clustering
```bash
python3 deeplab/scannet_uhlemeyer.py with subset=$SCENE pretrained_model=$MODEL pred_name=pred uncert_name=maxlogit-pp eps=3.5 min_samples=10
```
3. train the segmentation model and run inference with the new model
```bash
python deeplab/scannet_adaptation.py with subset=$SCENE and pretrained_model=$MODEL pseudolabels=uhlemeyer<id>
python deeplab/scannet_adaptedinference.py with training=<id from above> subset=$SCENE
```
  </details>

<details>
  <summary>our approach to SCIM</summary>

1. run inference
```bash
python deeplab/scannet_inference.py with subset=$SCENE and pretrained_model=$MODEL
```
2. run mapping (for flexibility, we run semantic mapping and uncertainty mapping separately)
```bash
roslaunch panoptic_mapping_utils scannnet_mapping.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
roslaunch panoptic_mapping_utils scannnet_uncertmap.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
```
3. render the maps
```bash
panoptic_mapping_utils scannnet_predrender.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
panoptic_mapping_utils scannnet_voxelidrender.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
panoptic_mapping_utils scannnet_uncertrender.launch scene:=$SCENE model:=$MODEL inference_path:=<folder for outputs>/scannet_inference/$SCENE/$MODEL
```
4. get the geometric features (we run 3DSmoothNet in a singularity container)
```bash
singularity run --nv --bind $OUTPUTS/$SCENE/$MODEL/point_cloud_0.ply:/pc.ply --bind $OUTPUTS/$SCENE/$MODEL/smoothnet:/output --bind $SMOOTHNET_DATA/evaluate:/3DSmoothNet/data/evaluate --bind $SMOOTHNET_DATA/logs:/3DSmoothNet/logs --bind $SMOOTHNET_DATA/preprocessed:/preprocessed smoothnet.simg
roslaunch panoptic_mapping_utils scannnet_geofeatures.launch scene:=$SCENE model:=$MODEL
```
5. run parameter optimisation and clustering (here we combine segmentation feautures, geometric features, and DINO; see the `deeplab/` folder for different scripts combining different features)
```bash
python3 deeplab/scannet_segandgeoanddino.py best_hdbscan  with subset=$SCENE pretrained_model=$MODEL n_calls=200 cluster_selection_method=eom
```
6. combine clustering and mapping into pseudolabels (`outlier` needs to be adjusted dependent on the clustering above)
```bash
python deeplab/pseudolabel.py with subset=$SCENE and pretrained_model=$MODEL outlier=segandgeoanddinohdbscan<id>
```
7. train the segmentation model and run inference with the new model
```bash
python deeplab/scannet_adaptation.py with subset=$SCENE and pretrained_model=$MODEL pseudolabels=merged-pseudolabel-pred-segandgeoanddinohdbscan<id>
python deeplab/scannet_adaptedinference.py with training=<id from above> subset=$SCENE
```
 </details>

# Installation

We offer a [dockerfile](https://github.com/hermannsblum/scim/blob/main/Dockerfile) that installs the whole code-base into a container. To install individual parts, see below:

## Clustering & Learning

This part is implemented in python. To install it, run:
```bash
git clone https://github.com/hermannsblum/scim.git
cd scim && python -m pip install -e .
```

## Mapping

For mapping, we rely on an [existing mapping framework](https://github.com/ethz-asl/panoptic_mapping). This is implemented in ROS.

First, create a catkin workspace:
```bash
sudo apt-get install python3-catkin-tools
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
catkin config --extend /opt/ros/noetic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
catkin config --merge-devel
```

Then install the framework into a catkin workspace:
```bash
wstool init \
  && git clone --branch hermann-devel https://github.com/ethz-asl/panoptic_mapping.git \
  && wstool merge panoptic_mapping/panoptic_mapping_https.rosinstall \
  && wstool update -j8 \
  && catkin build panoptic_mapping_utils point_cloud_io
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
The `<folder for outputs>` is also the one that should be used in the `inference_path:=` argument to the roslaunch files.


Experimental logs are stored with [sacred](https://github.com/IDSIA/sacred). If instead of tracking them in a folder, you want to track them in a database, please add the following lines to `settings.py`:
```python
EXPERIMENT_DB_HOST =
EXPERIMENT_DB_USER =
EXPERIMENT_DB_PWD =
EXPERIMENT_DB_NAME =
```

## ScanNet

Unfortunately, the code does not yet directly download from ScanNet. Therefore, first download the relevant scenes as described [here](https://github.com/ScanNet/ScanNet) and then put them in a zip archive called `valscans.zip` that you store in `~/tensorflow_datasets/downloads/manual/valscans.zip`. TFDS will then automatically extract, resize, and load the scenes.
