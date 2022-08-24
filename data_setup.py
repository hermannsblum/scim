import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import torchvision
import tensorflow_datasets as tfds
import semseg_density.data.coco_segmentation
import semseg_density.data.nyu_depth_v2
import semseg_density.data.scannet

from semseg_density.model.refinenet import rf_lw50, rf_lw101
from semseg_density.gdrive import load_gdrive_file

tfds.load('coco/2017_panoptic')
tfds.load('nyu_depth_v2_labeled/labeled')
tfds.load('nyu_depth_v2_labeled/no-bag')
tfds.load('nyu_depth_v2_labeled/no-book')
tfds.load('nyu_depth_v2_labeled/book')
tfds.load('nyu_depth_v2_labeled/no-tv')
tfds.load('nyu_depth_v2_labeled/tv')
tfds.load('coco_segmentation')
tfds.load('coco_segmentation/nyu')
tfds.load('coco_segmentation/nyu-nobag')
tfds.load('coco_segmentation/nyu-nobook')
tfds.load('coco_segmentation/nyu-notv')
tfds.load('coco_segmentation/10-subset')
tfds.load('scan_net/25k')
tfds.load('scan_net/no-pillow-refridgerator-television')
tfds.load('scan_net/no-lamp')
tfds.load('scan_net/0to9')

rf_lw101(40)
rf_lw50(40)
model = torchvision.models.segmentation.deeplabv3_resnet101(
    pretrained=False,
    pretrained_backbone=True,
    progress=True,
    num_classes=40,
    aux_loss=None)
