# SCIM: Simultaneous Clustering, Inference, and Mapping for Open-World Semantic Scene Understanding

[[paper](https://arxiv.org/abs/2206.10670)]


This repository provides our evluation data, pretrained models, and implementations of the tested methods.

# Model Checkpoints

To get started on experiments easily, we provide DeepLabv3+ checkpoints trained on first COCO and then ScanNet, but excluding different classes:

<table>
<tr><th>Outlier Class</th><th>ScanNet Validation</th><th>Checkpoint</th></tr>
<tr><td>television</td><td></td><td><a target="_blank" href="https://zenodo.org/record/6840795/files/deeplab_no_tv.pth?download=1">download</a></td></tr>
<tr><td>books + bookshelf</td><td></td><td><a target="_blank" href="https://zenodo.org/record/6840795/files/deeplab_no_books.pth?download=1">download</a></td></tr>
<tr><td>towel</td><td></td><td><a target="_blank" href="https://zenodo.org/record/6840795/files/deeplab_no_towel.pth?download=1">download</a></td></tr>
</table>

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

## Nakajima

1. run inference  
```bash 
python deeplab/scannet_inference.py with subset=$SCENE and pretrained_model=$MODEL
```
2. run mapping (for flexibility, we run semantic mapping and uncertainty mapping separately)
```bash
roslaunch panoptic_mapping_utils scannnet_mapping.launch scene:=$SCENE model:=$MODEL
roslaunch panoptic_mapping_utils scannnet_uncertmap.launch scene:=$SCENE model:=$MODEL
```
3. render the maps
```bash
panoptic_mapping_utils scannnet_predrender.launch scene:=$SCENE model:=$MODEL
panoptic_mapping_utils scannnet_voxelidrender.launch scene:=$SCENE model:=$MODEL
panoptic_mapping_utils scannnet_uncertrender.launch scene:=$SCENE model:=$MODEL
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

## Uhlemeyer

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

## our approach to SCIM


1. run inference  
```bash 
python deeplab/scannet_inference.py with subset=$SCENE and pretrained_model=$MODEL
```
2. run mapping (for flexibility, we run semantic mapping and uncertainty mapping separately)
```bash
roslaunch panoptic_mapping_utils scannnet_mapping.launch scene:=$SCENE model:=$MODEL
roslaunch panoptic_mapping_utils scannnet_uncertmap.launch scene:=$SCENE model:=$MODEL
```
3. render the maps
```bash
panoptic_mapping_utils scannnet_predrender.launch scene:=$SCENE model:=$MODEL
panoptic_mapping_utils scannnet_voxelidrender.launch scene:=$SCENE model:=$MODEL
panoptic_mapping_utils scannnet_uncertrender.launch scene:=$SCENE model:=$MODEL
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

# Installation


