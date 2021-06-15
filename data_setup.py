import tensorflow_datasets as tfds
import fastscnn.data.coco_segmentation
import fastscnn.data.nyu_depth_v2

from fastscnn.gdrive import load_gdrive_file

tfds.load('coco/2017_panoptic')
tfds.load('nyu_depth_v2_labeled/labeled')
tfds.load('coco_segmentation')

load_gdrive_file('15vSz0eBSjN4K1b3DvKZ1PQTtW4cY-_2W', ending='pth')
load_gdrive_file('1zLhaoGjnU5nDyAfcBnPss6NFBKuo9iuE', ending='pth')
load_gdrive_file('1yb7izcZyyjL3Bm-Y1vNoUyHBWOhqUen_', ending='pth')
load_gdrive_file('1QvES0v_oHTgmx0sxZsAjvCWQvxS3Z6DO', ending='pth')
load_gdrive_file('1EnMk9YWPGET2sBBgjni7deadYOYPqn0O', ending='pth')
load_gdrive_file('1Fk9efJuv07bnNy91l1xzjEFtQ_qyyufR', ending='pth')

