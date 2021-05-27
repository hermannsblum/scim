import tensorflow_datasets as tfds
import fastscnn.data.coco_segmentation
import fastscnn.data.nyu_depth_v2

tfds.load('coco/2017_panoptic')
tfds.load('nyu_depth_v2_labeled/labeled')
tfds.load('coco_segmentation')
