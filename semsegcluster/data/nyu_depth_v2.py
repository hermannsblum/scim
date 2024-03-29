# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NYU Depth V2 Dataset."""

import os

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_datasets.public_api as tfds

_CITATION = """\
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
@inproceedings{icra_2019_fastdepth,
  author    = {Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne},
  title     = {FastDepth: Fast Monocular Depth Estimation on Embedded Systems},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2019}
}
"""

_DESCRIPTION = """\
The NYU-Depth V2 data set is comprised of video sequences from a variety of
indoor scenes as recorded by both the RGB and Depth cameras from the
Microsoft Kinect. It includes a labelled subset of frames with semantic and
instance annotations.
"""

TRAINING_LABEL_MAP = [
    256, 40, 40, 3, 22, 5, 40, 12, 38, 40, 40, 2, 39, 40, 40, 26, 40, 24, 40, 7,
    40, 1, 40, 40, 34, 38, 29, 40, 8, 40, 40, 40, 40, 38, 40, 40, 14, 40, 38,
    40, 40, 40, 15, 39, 40, 30, 40, 40, 39, 40, 39, 38, 40, 38, 40, 37, 40, 38,
    38, 9, 40, 40, 38, 40, 11, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 38, 13, 40, 40, 6, 40, 23, 40, 39, 10, 16, 40, 40, 40, 40, 38, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 39, 40, 40, 40, 40, 39, 38, 40, 40,
    40, 40, 40, 40, 18, 40, 40, 19, 28, 33, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    38, 27, 36, 40, 40, 40, 40, 21, 40, 20, 35, 40, 40, 40, 40, 40, 40, 40, 40,
    38, 40, 40, 40, 4, 32, 40, 40, 39, 40, 39, 40, 40, 40, 40, 40, 17, 40, 40,
    25, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 39, 38, 38, 40, 40, 39, 40, 39, 40, 38, 39, 38, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 39, 40, 38, 40, 40, 38, 38, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 39, 40, 40, 40, 38, 40, 40, 39, 40, 40, 38, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 31, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 38,
    39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 39, 40, 40, 39, 40, 40,
    40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 38, 39, 40, 40, 40, 40, 40, 40, 38,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 39, 40, 40, 40, 40, 40, 40,
    40, 39, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 38, 40, 39, 40, 40, 40, 39,
    39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 39, 39, 40, 40, 39, 39, 40, 40, 40, 40, 38, 40, 40, 38,
    39, 39, 40, 39, 40, 39, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39,
    40, 38, 40, 39, 40, 40, 40, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 39, 39,
    40, 40, 38, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 40, 40, 40,
    40, 39, 40, 40, 40, 40, 40, 39, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 39, 38, 40, 40, 40, 40, 40, 40, 40, 39, 38, 39,
    40, 38, 39, 40, 39, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40,
    40, 40, 38, 40, 40, 39, 40, 40, 40, 39, 40, 38, 40, 40, 40, 40, 40, 40, 40,
    40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39,
    38, 40, 40, 38, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 38, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 38, 38, 40, 40, 40, 38, 40, 40,
    40, 38, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 38, 39,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 39, 40, 40, 40, 40, 38, 38, 40,
    40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40,
    40, 39, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 40, 40,
    40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 39, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40,
    40, 40, 39, 40, 40, 38, 40, 39, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 38,
    40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40,
    40
]

# according to https://github.com/ankurhanda/SceneNetv1.0/
TRAINING_LABEL_NAMES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
    'curtain', 'dresser', 'pilow', 'mirror', 'floor mat', 'clothes', 'ceiling',
    'books', 'refridgerator', 'television', 'paper', 'towel', 'shower-curtain',
    'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp',
    'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop'
]


class NyuDepthV2Config(tfds.core.BuilderConfig):

  def __init__(self, labeled=False, classes=range(40), min_classes=[], **kwargs):
    super().__init__(version='1.1.0', **kwargs)
    # classes that are allowed to be in the set
    self.classes = list(classes)
    # classes that must be in the set
    self.min_classes = list(min_classes)
    self.labeled = labeled


class NyuDepthV2Labeled(tfds.core.GeneratorBasedBuilder):
  """NYU Depth V2 Dataset."""

  VERSION = tfds.core.Version('1.1.0')

  BUILDER_CONFIGS = [
      NyuDepthV2Config(
          name='bag',
          description='Subset of frames with only bags.',
          labeled=True,
          min_classes=[i for i in range(40) if TRAINING_LABEL_NAMES[i] == 'bag'],
      ),
      NyuDepthV2Config(
          name='no-bag',
          description='Subset of frames without any bags.',
          labeled=True,
          classes=[i for i in range(40) if TRAINING_LABEL_NAMES[i] != 'bag'],
      ),
      NyuDepthV2Config(
          name='book',
          description='Subset of frames with only books.',
          labeled=True,
          min_classes=[i for i in range(40) if TRAINING_LABEL_NAMES[i] == 'books'],
      ),
      NyuDepthV2Config(
          name='no-book',
          description='Subset of frames without any books.',
          labeled=True,
          classes=[i for i in range(40) if TRAINING_LABEL_NAMES[i] != 'books'],
      ),
      NyuDepthV2Config(
          name='tv',
          description='Subset of frames with only TVs.',
          labeled=True,
          min_classes=[i for i in range(40) if TRAINING_LABEL_NAMES[i] == 'television'],
      ),
      NyuDepthV2Config(
          name='no-tv',
          description='Subset of frames without any TVs.',
          labeled=True,
          classes=[i for i in range(40) if TRAINING_LABEL_NAMES[i] != 'television'],
      ),
      NyuDepthV2Config(
          name='labeled',
          description=
          'Subset of frames labelled for semantic and instance segmentation.',
          labeled=True,
      ),
      NyuDepthV2Config(
          name='depth',
          description='Full dataset with rgb and depth images.',
          labeled=False,
      ),
  ]

  def _info(self):
    features = {
        'image': tfds.features.Image(shape=(480, 640, 3)),
        'depth': tfds.features.Tensor(shape=(480, 640), dtype=tf.float16),
    }
    if self.builder_config.labeled:
      features.update({
          'depth': tfds.features.Tensor(shape=(480, 640), dtype=tf.float32),
          'accelData': tfds.features.Tensor(shape=(4,), dtype=tf.float32),
          'instances': tfds.features.Tensor(shape=(480, 640), dtype=tf.uint8),
          'labels': tfds.features.Tensor(shape=(480, 640), dtype=tf.uint8),
          'scene': tf.string,
          'sceneType': tf.string,
          'name': tf.string,
      })
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        supervised_keys=('image',
                         'labels' if self.builder_config.labeled else 'depth'),
        homepage='https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    urls = {
        'fastdepth':
            'http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz',
        'labeled':
            'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat',
    }

    if self.builder_config.labeled:
      extracted = dl_manager.download_and_extract(urls['labeled'])
      return [
          tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                   gen_kwargs={
                                       'data_path': extracted,
                                   })
      ]
    extracted = dl_manager.download_and_extract(urls['fastdepth'])
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'data_path': os.path.join(extracted, 'nyudepthv2', 'train')
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'data_path': os.path.join(extracted, 'nyudepthv2', 'val')
            },
        ),
    ]

  def _generate_examples(self, data_path):
    """Yields examples."""
    h5py = tfds.core.lazy_imports.h5py
    label_map = np.array(TRAINING_LABEL_MAP) - 1
    if self.builder_config.labeled:
      with h5py.File(data_path, 'r') as f:
        for i in range(f['accelData'].shape[1]):
          labels = label_map[np.transpose(
              f['labels'][i]).astype('uint16')].astype('uint8')
          # check that image only contains allowed labels
          valid = True
          labels_in_this_frame = np.unique(labels).tolist()
          for l in labels_in_this_frame:
            if l == 255:
              # 255 is unlabeled
              continue
            if l not in self.builder_config.classes:
              valid = False
              break
          for l in self.builder_config.min_classes:
            if l not in labels_in_this_frame:
              valid = False
          if not valid:
            continue

          scene = f[f['scenes'][0, i]][:, 0].tobytes().decode('ascii')
          yield i, {
              'accelData':
                  f['accelData'][:, i].astype('float32'),
              'depth':
                  np.transpose(f['depths'][i]).astype('float32'),
              'image':
                  np.transpose(f['images'][i]).astype('uint8'),
              'instances':
                  np.transpose(f['instances'][i]).astype('uint8'),
              'labels':
                  labels,
              'scene':
                  scene,
              'sceneType':
                  f[f['sceneTypes'][0, i]][:, 0].tobytes().decode('ascii'),
              'name': f'{i:06d}',
          }
    else:
      for directory in tf.io.gfile.listdir(data_path):
        for file_name in tf.io.gfile.listdir(os.path.join(data_path,
                                                          directory)):
          with h5py.File(os.path.join(data_path, directory, file_name),
                         'r') as f:
            yield directory + '_' + file_name, {
                'image': np.transpose(f['rgb'], (1, 2, 0)),
                'depth': f['depth'][:].astype('float16')
            }


NYU40_COLORS = [
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),
    (247, 182, 210),  # desk
    (66, 188, 102),
    (219, 219, 141),  # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14),  # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),  # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),
    (227, 119, 194),  # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  # otherfurn
    (100, 85, 144)
]
