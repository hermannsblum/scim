import resource
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import tensorflow_datasets as tfds
import semseg_density.data.coco_segmentation
import semseg_density.data.nyu_depth_v2
import semseg_density.data.scannet

from semseg_density.model.refinenet import rf_lw50, rf_lw101
from semseg_density.gdrive import load_gdrive_file

tfds.load('coco/2017_panoptic')
tfds.load('nyu_depth_v2_labeled/labeled')
tfds.load('nyu_depth_v2_labeled/no-bag')
tfds.load('coco_segmentation')
tfds.load('scan_net/25k')
tfds.load('scan_net/no-pillow-refridgerator-television')
tfds.load('scan_net/no-lamp')
tfds.load('scan_net/0to9')

load_gdrive_file('15vSz0eBSjN4K1b3DvKZ1PQTtW4cY-_2W', ending='pth')
load_gdrive_file('1zLhaoGjnU5nDyAfcBnPss6NFBKuo9iuE', ending='pth')
load_gdrive_file('1yb7izcZyyjL3Bm-Y1vNoUyHBWOhqUen_', ending='pth')
load_gdrive_file('1QvES0v_oHTgmx0sxZsAjvCWQvxS3Z6DO', ending='pth')
load_gdrive_file('1EnMk9YWPGET2sBBgjni7deadYOYPqn0O', ending='pth')
load_gdrive_file('1Fk9efJuv07bnNy91l1xzjEFtQ_qyyufR', ending='pth')
load_gdrive_file('1mjQPljaUqd6g2cjkfQIm8QbXOd-Qg6NU', ending='pth')
load_gdrive_file('1uHI2pLJCcSVZaWSTVWFQ8-maHg9o3d7o', ending='pth')

rf_lw101(40)
rf_lw50(40)
