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

_CITATION = """\
"""

_DESCRIPTION = """\
25k sampled  images from scannet
"""


class ScanNetConfig(tfds.core.BuilderConfig):

  def __init__(self, classes=range(40), **kwargs):
    super().__init__(version='1.0.0', **kwargs)
    self.classes = list(classes)


class ScanNet(tfds.core.GeneratorBasedBuilder):
  """NYU Depth V2 Dataset."""

  VERSION = tfds.core.Version('1.0.0')

  BUILDER_CONFIGS = [
      ScanNetConfig(
          name='25k',
          description='all subsampled images',
      ),
  ]

  def _info(self):
    features = {
        'image': tfds.features.Image(shape=(480, 640, 3)),
        'scene': tf.string,
        'labels': tfds.features.Tensor(shape=(480, 640), dtype=tf.uint8),
    }
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
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
    }

    extracted = dl_manager.download_and_extract(urls['subsampled'])

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'data_path': os.path.join(extracted, 'scannet_frames_25k')
            },
        ),
    ]

  def _generate_examples(self, data_path):
    """Yields examples."""
    for scene_dir in tf.io.gfile.listdir(data_path):
      for file_name in tf.io.gfile.listdir(
          os.path.join(data_path, scene_dir, 'color')):
        index = file_name.split('.')[0]
        labels = cv2.imread(
            os.path.join(data_path, scene_dir, 'label', f'{index}.png'),
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
        labels = resize_with_crop(np.expand_dims(labels, -1), (480, 640), method='nearest').numpy()[..., 0]
        image = cv2.imread(os.path.join(data_path, scene_dir, 'color', file_name))[..., ::-1]
        image = resize_with_crop(image, (480, 640)).numpy().astype('uint8')
        yield f'{scene_dir}_{index}', {
            'image': image,
            'labels': labels,
            'scene': scene_dir,
        }
