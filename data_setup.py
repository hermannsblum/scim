import tensorflow_datasets as tfds
import fastscnn.data.coco_segmentation

tfds.load('coco/2017_panoptic')
tfds.load('coco_segmentation')
