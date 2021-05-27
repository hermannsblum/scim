import tensorflow_datasets as tfds
import fastscnn.data.coco_segmentation
import fastscnn.data.nyu_depth_v2

from fastscnn.gdrive import load_gdrive_file

tfds.load('coco/2017_panoptic')
tfds.load('nyu_depth_v2_labeled/labeled')
tfds.load('coco_segmentation')

load_gdrive_file('15vSz0eBSjN4K1b3DvKZ1PQTtW4cY-_2W', ending='pth')
