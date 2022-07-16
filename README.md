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

# Installation


