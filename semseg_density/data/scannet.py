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
import cv2

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_datasets.public_api as tfds

from semseg_density.data.images import resize_with_crop
from semseg_density.data.nyu_depth_v2 import TRAINING_LABEL_NAMES

_CITATION = """\
"""

_DESCRIPTION = """\
Various Splits of Data from the ScanNet dataset.
"""


class ScanNetConfig(tfds.core.BuilderConfig):

  def __init__(self, scene, classes=range(40), subsampling=None, **kwargs):
    super().__init__(version='1.0.0', **kwargs)
    self.classes = list(classes)
    assert isinstance(scene, int) or scene in ('25k', '0to9')
    self.scene = scene
    self.subsampling = subsampling


class ScanNet(tfds.core.GeneratorBasedBuilder):
  """NYU Depth V2 Dataset."""

  VERSION = tfds.core.Version('1.0.0')

  BUILDER_CONFIGS = [
      ScanNetConfig(name='25k',
                    description='all subsampled images',
                    scene='25k'),
      ScanNetConfig(
          name='scene0',
          description='full scene 0',
          scene=0,
      ),
      ScanNetConfig(
          name='0to9',
          description='All frames from scene 0 to 9',
          scene='0to9',
          subsampling=20,
      ),
      ScanNetConfig(
          name='no-pillow-refridgerator-television',
          scene='25k',
          description=
          'Only contains images without pillow, refridgerator, or television.',
          classes=[
              i for i in range(40)
              if not TRAINING_LABEL_NAMES[i] in ('pilow', 'refridgerator',
                                                 'television')
          ]),
      ScanNetConfig(
          name='no-lamp',
          scene='25k',
          description='Only contains images without lamps.',
          classes=[
              i for i in range(40) if not TRAINING_LABEL_NAMES[i] in ('lamp')
          ]),
  ]

  def _info(self):
    features = {
        'image': tfds.features.Image(shape=(480, 640, 3)),
        'scene': tf.string,
        'name': tf.string,
        'labels': tfds.features.Tensor(shape=(480, 640), dtype=tf.uint8),
    }
    if self.builder_config.scene == '0to9':
      features['instances'] = tfds.features.Tensor(shape=(480, 640),
                                                   dtype=tf.uint8)
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        # only 25k should be shuffled
        disable_shuffling=self.builder_config.scene != '25k',
        features=tfds.features.FeaturesDict(features),
        supervised_keys=('image', 'labels'),
        homepage='http://www.scan-net.org',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    urls = {
        'subsampled':
            'https://drive.google.com/uc?export=download&id=1UFjiMx_yuA94duiOIYgj355f79vkopFk',
        'full':
            'https://drive.google.com/uc?export=download&id=1E6D0pLqxUHDZCMoStNXlYOHfMekebdEr',
        '0to9':
            'https://drive.google.com/uc?export=download&id=1rI6t0DYQMENS3dS2PTxulvSKa39Tf1Hq',
    }

    if self.builder_config.scene == '25k':
      # take data from subsampled dataset
      extracted = dl_manager.download_and_extract(urls['subsampled'])
      return [
          tfds.core.SplitGenerator(
              name=tfds.Split.TRAIN,
              gen_kwargs={
                  'data_path': os.path.join(extracted, 'scannet_frames_25k')
              },
          ),
      ]
    elif self.builder_config.scene == '0to9':
      extracted = dl_manager.download_and_extract(urls['0to9'])
      return [
          tfds.core.SplitGenerator(
              name=tfds.Split.TRAIN,
              gen_kwargs={'data_path': os.path.join(extracted, '0to9')},
          ),
      ]
    extracted = dl_manager.download_and_extract(urls['full'])
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={'data_path': os.path.join(extracted, 'scannet')})
    ]

  def _generate_examples(self, data_path):
    """Yields examples."""
    subsampler = 0
    for scene_dir in sorted(tf.io.gfile.listdir(data_path)):
      if isinstance(
          self.builder_config.scene, int
      ) and not scene_dir.startswith(f'scene{self.builder_config.scene:4d}'):
        continue
      for file_name in sorted(
          tf.io.gfile.listdir(os.path.join(data_path, scene_dir, 'color'))):
        subsampler += 1
        if (self.builder_config.subsampling is not None) and (
            subsampler % self.builder_config.subsampling != 0):
          continue
        index = int(file_name.split('.')[0])
        if os.path.exists(os.path.join(data_path, scene_dir, 'label-filt')):
          raw_labels = cv2.imread(
              os.path.join(data_path, scene_dir, 'label-filt', f'{index}.png'),
              cv2.IMREAD_ANYDEPTH)
          # apply mapping
          labels = SCANNET_TO_NYU40[raw_labels]
        else:
          assert os.path.exists(os.path.join(data_path, scene_dir, 'label'))
          labels = cv2.imread(
              os.path.join(data_path, scene_dir, 'label', f'{index:06d}.png'),
              cv2.IMREAD_ANYDEPTH)
          # labels are alrady mapped to NYU, but not our format
          labels[labels == 0] = 256
          labels = (labels - 1).astype('uint8')
        contained_labels = np.unique(labels)
        # check that image only contains allowed labels
        valid = True
        for l in contained_labels.tolist():
          if l == 255:
            # 255 is unlabeled
            continue
          if l not in self.builder_config.classes:
            valid = False
            break
        if not valid:
          continue
        # reshape to common size
        labels = resize_with_crop(np.expand_dims(labels, -1), (480, 640),
                                  method='nearest').numpy()[..., 0]

        if 'instances' in self.info.features:
          assert os.path.exists(
              os.path.join(data_path, scene_dir, 'instance-filt'))
          instances = cv2.imread(
              os.path.join(data_path, scene_dir, 'instance-filt',
                           f'{index}.png'), cv2.IMREAD_ANYDEPTH)
          # reshape to common size
          instances = resize_with_crop(np.expand_dims(instances, -1), (480, 640),
                                    method='nearest').numpy()[..., 0]

        image = cv2.imread(
            os.path.join(data_path, scene_dir, 'color', file_name))[..., ::-1]
        image = resize_with_crop(image, (480, 640)).numpy().astype('uint8')

        # create an integer key, required if shuffling is disabled
        scenenum = int(scene_dir.split('_')[0][5:])
        trajectorynum = int(scene_dir.split('_')[1])
        key = scenenum * 100000 + trajectorynum * 10000 + index
        features = {
            'image': image,
            'labels': labels,
            'scene': scene_dir,
            'name': f'{scene_dir}_{index:06d}',
        }
        if 'instances' in self.info.features:
          features['instances'] = instances
        yield key, features


SCANNET_TO_NYU40 = np.array([
    255, 0, 4, 1, 6, 7, 5, 2, 14, 13, 4, 3, 255, 17, 33, 10, 8, 32, 9, 39, 255,
    15, 22, 4, 6, 38, 28, 23, 34, 2, 255, 26, 20, 24, 31, 11, 16, 255, 39, 17,
    39, 21, 35, 38, 6, 6, 39, 36, 39, 39, 39, 38, 29, 39, 39, 27, 38, 38, 37,
    39, 255, 39, 37, 39, 39, 39, 39, 38, 38, 37, 38, 18, 39, 39, 4, 2, 39, 39,
    37, 25, 30, 38, 39, 39, 37, 38, 12, 38, 39, 37, 38, 38, 39, 39, 255, 37, 38,
    38, 39, 38, 39, 39, 39, 39, 38, 37, 39, 37, 6, 255, 38, 39, 39, 255, 255,
    39, 38, 37, 39, 39, 37, 39, 38, 39, 255, 37, 38, 255, 0, 38, 39, 39, 39, 39,
    39, 39, 37, 255, 39, 39, 19, 37, 39, 38, 39, 37, 37, 255, 39, 255, 255, 255,
    37, 38, 37, 38, 37, 39, 255, 11, 37, 7, 255, 39, 255, 23, 39, 39, 39, 39,
    39, 255, 255, 255, 39, 255, 255, 38, 255, 34, 39, 255, 39, 255, 255, 39,
    255, 255, 37, 39, 255, 37, 255, 37, 39, 39, 255, 255, 255, 255, 255, 255,
    39, 255, 39, 255, 255, 255, 39, 255, 255, 255, 39, 38, 39, 255, 39, 255,
    255, 255, 39, 39, 6, 255, 255, 39, 39, 39, 39, 39, 39, 39, 37, 38, 38, 37,
    255, 255, 39, 255, 255, 255, 39, 255, 255, 39, 255, 39, 255, 255, 39, 255,
    255, 255, 255, 255, 255, 39, 255, 255, 255, 39, 255, 255, 39, 39, 255, 255,
    255, 39, 255, 255, 255, 255, 255, 255, 7, 255, 255, 255, 39, 39, 38, 39, 39,
    255, 39, 255, 255, 38, 255, 39, 255, 255, 255, 255, 255, 39, 39, 255, 39,
    37, 255, 255, 39, 37, 255, 38, 255, 255, 255, 255, 39, 255, 255, 255, 39,
    255, 255, 39, 255, 39, 255, 39, 255, 39, 38, 255, 255, 255, 255, 39, 39,
    255, 255, 255, 255, 255, 255, 39, 255, 255, 37, 255, 255, 39, 39, 255, 255,
    255, 255, 255, 255, 255, 38, 255, 39, 26, 255, 255, 255, 39, 255, 38, 255,
    39, 39, 255, 255, 255, 39, 255, 39, 255, 255, 255, 255, 255, 39, 37, 255,
    255, 255, 255, 255, 2, 39, 39, 255, 38, 255, 255, 39, 255, 255, 37, 255, 39,
    255, 39, 255, 39, 255, 255, 255, 255, 255, 255, 39, 255, 38, 38, 255, 255,
    255, 39, 255, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 39, 255, 39, 39, 39, 255, 255, 255, 36, 255, 255, 255, 255, 255,
    255, 255, 37, 255, 39, 255, 39, 255, 255, 255, 255, 255, 255, 39, 255, 39,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 36, 255, 255, 255, 38, 255, 255, 255, 39, 255, 255, 255, 255,
    255, 3, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 36, 255, 255,
    255, 255, 255, 255, 37, 255, 255, 255, 255, 39, 255, 255, 255, 255, 39, 255,
    38, 255, 255, 255, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 28,
    255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 39, 255, 255, 255, 255, 39,
    255, 255, 255, 255, 39, 39, 39, 255, 255, 38, 255, 255, 7, 36, 39, 39, 255,
    255, 255, 255, 255, 255, 255, 255, 38, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 39, 39, 39, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255,
    255, 255, 39, 255, 37, 255, 255, 39, 255, 255, 255, 255, 255, 255, 255, 255,
    39, 255, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255, 7, 255, 255,
    255, 255, 255, 255, 255, 28, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 28, 255, 255, 255, 255, 255, 255, 255, 255,
    39, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 39, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 39,
    255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255, 255, 39, 255,
    255, 255, 37, 255, 255, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 39, 255, 39, 34, 39, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 38, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 3, 255, 255, 255,
    255, 255, 255, 255, 39, 255, 39, 255, 255, 255, 255, 39, 255, 39, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 39, 39, 2, 39, 39, 255, 39, 255, 255, 38,
    255, 255, 255, 255, 255, 39, 39, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 38, 255,
    255, 255, 255, 255, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 4, 255, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 39, 255, 255, 39, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255,
    255, 255, 255, 0, 39, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 39, 255, 255,
    255, 37, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255, 255, 255,
    39, 255, 255, 39, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 23, 37, 255, 255, 255,
    255, 255, 255, 255, 255, 36, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 39, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 39, 38, 255, 255, 255, 255, 37, 255, 255, 39, 39, 255,
    255, 255, 255, 255, 255, 255, 255, 39, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 34, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 11, 255,
    255, 255, 255, 255, 255, 39, 2, 0, 39, 7, 37, 38, 39, 37, 39, 2, 39, 39, 39,
    38, 37, 37, 38, 39, 39, 39, 4, 39, 39, 39, 10, 39, 39, 3, 37, 6, 39, 39, 39,
    20, 39, 39, 39, 37, 38, 39, 0, 39, 39, 39, 7, 38, 39, 39, 39, 22, 39, 37,
    37, 39, 10, 37, 37, 39, 39, 39, 39, 21, 39, 38, 39, 39, 36, 39, 36, 39, 39,
    39, 39, 39, 36, 39, 39, 39, 38, 0, 38, 28, 39, 39, 39, 39, 39, 39, 36, 28,
    30, 39, 38, 32, 39, 39, 39, 39, 39, 39, 36, 39, 39, 19, 39, 39, 38, 28, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 14, 39, 39, 39, 39, 36, 39, 38, 4,
    39, 39, 39, 37, 39, 36, 39, 28, 39, 20, 39, 20, 39, 39, 39, 39, 39, 39, 37,
    39, 39, 5, 39, 39, 14, 39, 39, 39, 39, 39, 2, 39, 39, 39, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 14, 39, 39, 4, 39, 37, 38, 20, 39, 39, 7, 39, 1, 39, 3,
    39, 38, 39, 39, 39, 6, 36, 39
]).astype('uint8')
