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
    #assert isinstance(scene, int) or scene in ('25k', '0to9')
    self.scene = scene
    self.subsampling = subsampling


class ScanNet(tfds.core.GeneratorBasedBuilder):
  """NYU Depth V2 Dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  For val100: Download from https://drive.google.com/file/d/1H8_soxzY-5rs8khGv-vWmBuZFsMzpoMA/view?usp=sharing
  """

  VERSION = tfds.core.Version('1.0.0')

  BUILDER_CONFIGS = [
      ScanNetConfig(name='full', description='all images', scene='full'),
      ScanNetConfig(name='val100',
                    description='100 validation scenes',
                    scene='val100'),
      ScanNetConfig(name='25k',
                    description='all subsampled images',
                    scene='25k'),
      ScanNetConfig(
          name='0to9',
          description='Frames from scene 0 to 9',
          scene='0to9',
          subsampling=10,
      ),
      ScanNetConfig(
          name='no-towel',
          scene='25k',
          description='Only contains images without towels',
          classes=[
              i for i in range(40) if not TRAINING_LABEL_NAMES[i] in ('towel')
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
        'image':
            tfds.features.Image(shape=(480, 640, 3)),
        'scene':
            tf.string,
        'name':
            tf.string,
        'labels_nyu':
            tfds.features.Tensor(shape=(480, 640), dtype=tf.uint8),
        'labels_eigen12':
            tfds.features.Tensor(shape=(480, 640), dtype=tf.uint8),
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
    elif self.builder_config.scene == 'val100':
      extracted = dl_manager.extract(
          os.path.join(dl_manager.manual_dir, 'valscans.zip'))
      return [
          tfds.core.SplitGenerator(
              name=tfds.Split.VALIDATION,
              gen_kwargs={'data_path': os.path.join(extracted)},
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
      if scene_dir in VAL_100_SCANS and self.builder_config.scene != 'val100':
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
          nyu_labels = (SCANNET_TO_NYU40[raw_labels] - 1).astype('uint8')
          eigen_labels = (SCANNET_TO_EIGEN12[raw_labels] - 1).astype('uint8')
        else:
          assert os.path.exists(os.path.join(data_path, scene_dir, 'label'))
          nyu_labels = cv2.imread(
              os.path.join(data_path, scene_dir, 'label', f'{index:06d}.png'),
              cv2.IMREAD_ANYDEPTH)
          # labels are alrady mapped to NYU, but not our format
          eigen_labels = (NYU40_TO_EIGEN12[nyu_labels] - 1).astype('uint8')
          nyu_labels[nyu_labels == 0] = 256
          nyu_labels = (nyu_labels - 1).astype('uint8')
        contained_labels = np.unique(nyu_labels)
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
        nyu_labels = resize_with_crop(np.expand_dims(nyu_labels, -1),
                                      (480, 640),
                                      method='nearest').numpy()[..., 0]
        eigen_labels = resize_with_crop(np.expand_dims(eigen_labels, -1),
                                        (480, 640),
                                        method='nearest').numpy()[..., 0]

        if 'instances' in self.info.features:
          assert os.path.exists(
              os.path.join(data_path, scene_dir, 'instance-filt'))
          instances = cv2.imread(
              os.path.join(data_path, scene_dir, 'instance-filt',
                           f'{index}.png'), cv2.IMREAD_ANYDEPTH)
          # reshape to common size
          instances = resize_with_crop(np.expand_dims(instances, -1),
                                       (480, 640),
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
            'labels_nyu': nyu_labels,
            'labels_eigen12': eigen_labels,
            'scene': scene_dir,
            'name': f'{scene_dir}_{index:06d}',
        }
        if 'instances' in self.info.features:
          features['instances'] = instances
        yield key, features


# below IDs as sorted in the ScanNet label mapping file (some are double)
SCANNET_IDS = np.array([
    1, 2, 22, 3, 5, 1163, 16, 4, 56, 13, 15, 41, 26, 161, 19, 7, 9, 8, 10, 31,
    6, 14, 48, 28, 11, 18, 71, 21, 40, 52, 96, 22, 29, 49, 29, 23, 63, 24, 17,
    47, 32, 46, 65, 97, 34, 38, 33, 75, 36, 64, 32, 101, 130, 27, 44, 131, 55,
    42, 59, 159, 74, 82, 1164, 93, 77, 67, 128, 50, 35, 69, 100, 62, 105, 1165,
    165, 7, 5, 76, 230, 54, 125, 72, 68, 145, 157, 1166, 132, 1167, 232, 134,
    51, 250, 1168, 342, 89, 103, 99, 95, 154, 140, 1169, 193, 116, 202, 73, 78,
    1170, 79, 80, 141, 57, 102, 261, 118, 136, 98, 1171, 170, 1172, 1173, 79,
    221, 570, 138, 168, 276, 106, 214, 276, 323, 58, 86, 2, 399, 121, 185, 300,
    180, 163, 26, 66, 208, 112, 540, 395, 166, 122, 120, 107, 283, 88, 90, 177,
    1174, 562, 1175, 1156, 84, 104, 229, 70, 325, 169, 128, 331, 87, 488, 776,
    370, 191, 748, 242, 45, 417, 70, 188, 1176, 1177, 1178, 18, 110, 148, 63,
    155, 572, 1179, 28, 392, 1180, 609, 1181, 195, 581, 58, 1182, 1183, 139,
    1184, 1185, 156, 408, 213, 1186, 1187, 1188, 115, 1189, 304, 1190, 21, 312,
    233, 286, 264, 110, 1191, 356, 25, 750, 269, 307, 410, 730, 216, 1192, 119,
    682, 434, 126, 919, 85, 1193, 108, 135, 1194, 432, 53, 1195, 111, 305, 1125,
    13, 1196, 1197, 1198, 1199, 1200, 378, 591, 49, 92, 1098, 291, 1063, 107,
    1135, 189, 245, 194, 1201, 386, 1202, 857, 452, 1203, 346, 152, 83, 1204,
    726, 61, 39, 1117, 540, 1205, 415, 31, 1206, 153, 1207, 129, 220, 1208, 231,
    1209, 39, 1210, 117, 822, 238, 143, 1211, 228, 494, 226, 91, 1072, 435,
    1165, 345, 893, 621, 1212, 297, 1213, 1214, 1215, 529, 1216, 1217, 1218,
    1219, 1220, 525, 204, 693, 179, 1221, 1222, 1223, 1224, 1225, 1226, 1227,
    571, 1228, 556, 280, 1229, 1230, 1231, 1232, 746, 1233, 1234, 144, 282, 167,
    1235, 1236, 1237, 234, 563, 1238, 1239, 1240, 366, 816, 1241, 719, 284,
    1242, 247, 1243, 1244, 1245, 1246, 1247, 592, 385, 1248, 1249, 133, 301,
    1250, 379, 130, 1251, 450, 1252, 316, 1253, 385, 1254, 461, 1255, 1256, 599,
    281, 1257, 1258, 1259, 319, 1260, 28, 1261, 546, 1262, 1263, 1264, 1265,
    1266, 28, 1267, 1268, 1269, 188, 689, 1270, 1271, 1272, 354, 339, 1009,
    1273, 1274, 15, 1275, 361, 1276, 326, 1277, 1278, 116, 1184, 1279, 212,
    1280, 1281, 794, 1282, 955, 387, 69, 65, 523, 389, 29, 1283, 146, 130, 372,
    289, 440, 321, 976, 1256, 1284, 1285, 357, 1286, 1287, 365, 1288, 81, 1289,
    1290, 948, 174, 1028, 1291, 1292, 1005, 235, 1293, 1294, 1295, 1296, 1297,
    1298, 1299, 1300, 1301, 1051, 566, 1302, 1062, 1303, 1304, 1305, 1306, 298,
    1307, 1308, 1309, 43, 1310, 593, 1311, 1312, 749, 623, 1313, 99, 265, 1314,
    99, 1315, 448, 257, 1316, 786, 801, 972, 1317, 1318, 75, 657, 561, 93, 513,
    411, 1122, 922, 88, 518, 814, 1319, 1320, 649, 607, 819, 1321, 1322, 1204,
    227, 817, 130, 712, 1323, 1324, 673, 459, 643, 238, 765, 1008, 225, 1083,
    813, 138, 1145, 796, 1325, 363, 1174, 38, 1326, 997, 1327, 1328, 1329, 182,
    280, 1330, 1331, 1332, 1333, 939, 1334, 480, 907, 1335, 1336, 829, 947,
    1116, 599, 733, 123, 506, 569, 1337, 1338, 1314, 1339, 1340, 851, 142, 436,
    1341, 1342, 885, 693, 815, 401, 1343, 1184, 1344, 1345, 160, 1126, 1346,
    332, 397, 643, 551, 1163, 1347, 1348, 803, 484, 1349, 1350, 222, 1253, 1351,
    1352, 828, 1353, 612, 1354, 1355, 1301, 1356, 1199, 1357
])

EIGEN13_LABEL_NAMES = {
    1: 'bed',
    2: 'book',
    3: 'ceiling',
    4: 'chair',
    5: 'floor',
    6: 'furniture',
    7: 'object',
    8: 'picture',
    9: 'sofa',
    10: 'table',
    11: 'TV',
    12: 'wall',
    13: 'window',
}

EIGEN13_IDS = np.array([
    12, 4, 2, 5, 12, 7, 13, 10, 6, 7, 8, 3, 7, 12, 7, 6, 10, 6, 4, 7, 9, 7, 7,
    7, 1, 6, 7, 13, 7, 7, 6, 2, 6, 7, 6, 4, 7, 10, 7, 7, 7, 7, 7, 6, 6, 7, 11,
    6, 6, 7, 7, 7, 7, 6, 10, 7, 7, 7, 7, 6, 4, 7, 6, 7, 7, 6, 12, 7, 6, 7, 7, 7,
    7, 12, 6, 6, 12, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 12, 7, 7, 6, 7, 7, 7, 7, 7,
    6, 7, 7, 5, 6, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7,
    7, 7, 7, 7, 12, 7, 7, 12, 7, 7, 13, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 6, 7, 7, 7, 7, 6, 6, 7, 7, 7, 6, 7, 6, 7, 6, 7, 7, 12, 7, 6, 7, 7, 7, 7,
    7, 7, 10, 5, 6, 12, 7, 6, 7, 6, 6, 7, 7, 6, 7, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7,
    7, 7, 4, 7, 7, 7, 6, 7, 7, 8, 7, 7, 7, 7, 13, 7, 6, 7, 7, 6, 1, 7, 6, 7, 7,
    6, 6, 7, 7, 7, 7, 7, 7, 6, 7, 6, 10, 10, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 12, 7,
    7, 7, 6, 7, 7, 7, 7, 7, 6, 7, 6, 7, 12, 7, 6, 7, 7, 7, 6, 7, 6, 7, 7, 1, 7,
    6, 7, 7, 12, 7, 7, 7, 7, 7, 2, 7, 7, 7, 7, 7, 8, 7, 7, 6, 7, 7, 7, 7, 7, 7,
    7, 3, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 6, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 6, 7, 12, 6, 7, 7, 7, 7, 12, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 7,
    7, 7, 7, 6, 7, 7, 6, 7, 7, 7, 7, 8, 7, 7, 7, 6, 7, 7, 6, 4, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 6, 6, 6, 7, 7, 7, 6, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 6,
    7, 6, 7, 7, 7, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 6, 7, 7, 7,
    7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 9, 6, 7, 7, 6, 7, 7, 7, 6, 1, 7, 7, 7,
    7, 6, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 12, 7, 7, 7, 6, 12, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 12, 7, 7, 7, 7, 7, 12, 7, 4, 7, 7, 7, 6, 7, 7,
    6, 7, 4, 7, 6, 7, 7, 4, 7, 12, 7, 7, 7, 7, 7, 7, 7, 7, 5, 7, 7, 6, 1, 7, 10,
    7, 7, 7, 7, 7, 7, 7, 10, 7, 7, 7, 7
])

NYU40_IDS = np.array([
    1, 5, 23, 2, 8, 40, 9, 7, 39, 18, 11, 22, 29, 8, 40, 3, 14, 15, 5, 27, 6,
    34, 40, 35, 4, 10, 19, 16, 40, 30, 39, 23, 3, 40, 3, 5, 40, 7, 33, 37, 21,
    40, 40, 39, 32, 40, 25, 3, 17, 40, 21, 40, 40, 24, 7, 40, 28, 36, 40, 12, 5,
    40, 3, 40, 40, 39, 1, 40, 12, 38, 40, 38, 38, 1, 24, 3, 8, 40, 40, 40, 38,
    40, 39, 38, 40, 40, 40, 8, 38, 40, 39, 40, 38, 38, 38, 40, 39, 38, 38, 20,
    39, 38, 39, 40, 40, 38, 40, 26, 31, 38, 39, 40, 40, 40, 38, 40, 38, 40, 40,
    3, 26, 40, 37, 40, 40, 8, 40, 40, 8, 40, 38, 13, 5, 40, 40, 40, 40, 40, 40,
    29, 40, 40, 40, 29, 38, 40, 39, 38, 40, 40, 40, 39, 39, 40, 40, 40, 12, 38,
    39, 40, 39, 40, 40, 1, 40, 39, 40, 40, 40, 38, 40, 40, 7, 2, 39, 8, 40, 39,
    38, 10, 39, 40, 40, 39, 40, 38, 35, 40, 39, 38, 40, 40, 39, 38, 40, 40, 40,
    5, 40, 38, 40, 39, 40, 40, 11, 40, 40, 40, 40, 16, 40, 39, 40, 40, 39, 4,
    40, 39, 40, 40, 39, 39, 38, 40, 38, 40, 40, 40, 39, 40, 39, 7, 7, 40, 40,
    40, 40, 40, 40, 38, 40, 18, 40, 21, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38,
    38, 40, 40, 40, 40, 38, 40, 39, 40, 40, 40, 40, 38, 40, 1, 40, 40, 18, 39,
    29, 40, 40, 27, 40, 39, 40, 39, 40, 8, 40, 39, 18, 40, 38, 39, 40, 39, 40,
    40, 4, 40, 39, 37, 40, 1, 40, 40, 40, 40, 40, 23, 40, 38, 40, 38, 40, 11,
    38, 38, 39, 40, 40, 35, 40, 40, 40, 40, 22, 40, 39, 40, 40, 40, 40, 40, 37,
    40, 37, 40, 40, 40, 40, 39, 40, 40, 40, 40, 39, 40, 37, 40, 40, 40, 40, 40,
    40, 40, 39, 40, 1, 39, 29, 40, 40, 40, 8, 40, 40, 40, 38, 40, 38, 40, 40,
    40, 37, 40, 29, 3, 31, 40, 40, 39, 40, 40, 33, 40, 40, 40, 40, 35, 40, 40,
    40, 40, 37, 40, 40, 35, 20, 40, 40, 38, 40, 39, 29, 40, 39, 40, 40, 40, 40,
    11, 40, 40, 40, 39, 40, 40, 39, 5, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40,
    40, 39, 3, 15, 38, 40, 40, 39, 37, 40, 38, 39, 40, 40, 27, 40, 40, 40, 37,
    39, 40, 39, 40, 40, 40, 5, 40, 40, 38, 40, 40, 38, 40, 37, 40, 29, 40, 21,
    40, 39, 40, 24, 21, 40, 40, 40, 40, 40, 40, 40, 39, 38, 40, 40, 40, 35, 40,
    6, 39, 40, 40, 39, 40, 38, 40, 15, 4, 40, 40, 40, 40, 3, 29, 40, 40, 38, 39,
    38, 40, 40, 40, 40, 40, 40, 8, 40, 40, 40, 3, 1, 40, 40, 40, 40, 40, 40, 29,
    40, 40, 40, 39, 40, 40, 40, 40, 40, 35, 40, 40, 39, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 37, 40, 15, 40, 40, 1, 40, 40, 40, 40,
    37, 8, 40, 5, 40, 40, 38, 39, 40, 40, 39, 21, 5, 40, 3, 40, 40, 5, 40, 8,
    38, 40, 40, 40, 40, 40, 40, 40, 2, 40, 40, 39, 4, 40, 7, 29, 39, 40, 40, 40,
    40, 40, 7, 21, 37, 40, 40
])

SCANNET_TO_EIGEN13 = 256 * np.ones(max(SCANNET_IDS) + 1, dtype='uint16')
for i, scannet_id in enumerate(SCANNET_IDS):
  SCANNET_TO_EIGEN13[scannet_id] = EIGEN13_IDS[i]

SCANNET_TO_EIGEN12 = 256 * np.ones(max(SCANNET_IDS) + 1, dtype='uint16')
for i, scannet_id in enumerate(SCANNET_IDS):
  if EIGEN13_IDS[i] == 7:  # object
    SCANNET_TO_EIGEN12[scannet_id] = 256
  elif EIGEN13_IDS[i] > 7:
    SCANNET_TO_EIGEN12[scannet_id] = EIGEN13_IDS[i] - 1
  else:
    SCANNET_TO_EIGEN12[scannet_id] = EIGEN13_IDS[i]

SCANNET_TO_NYU40 = 256 * np.ones(max(SCANNET_IDS) + 1, dtype='uint16')
for i, scannet_id in enumerate(SCANNET_IDS):
  SCANNET_TO_NYU40[scannet_id] = NYU40_IDS[i]

NYU40_TO_EIGEN12 = 256 * np.ones(41, dtype='uint8')
for i, nyu_id in enumerate(NYU40_IDS):
  if EIGEN13_IDS[i] == 7:  # object
    NYU40_TO_EIGEN12[nyu_id] = 256
  elif EIGEN13_IDS[i] > 7:
    NYU40_TO_EIGEN12[nyu_id] = EIGEN13_IDS[i] - 1
  else:
    NYU40_TO_EIGEN12[nyu_id] = EIGEN13_IDS[i]
  # make sure that 38-40 map to 255 since some part of 39 also counts as 6 in eigen
  NYU40_TO_EIGEN12[38:] = 255

VAL_100_SCANS = [
    'scene0568_00', 'scene0568_01', 'scene0568_02', 'scene0304_00',
    'scene0488_00', 'scene0488_01', 'scene0412_00', 'scene0412_01',
    'scene0217_00', 'scene0019_00', 'scene0019_01', 'scene0414_00',
    'scene0575_00', 'scene0575_01', 'scene0575_02', 'scene0426_00',
    'scene0426_01', 'scene0426_02', 'scene0426_03', 'scene0549_00',
    'scene0549_01', 'scene0578_00', 'scene0578_01', 'scene0578_02',
    'scene0665_00', 'scene0665_01', 'scene0050_00', 'scene0050_01',
    'scene0050_02', 'scene0257_00', 'scene0025_00', 'scene0025_01',
    'scene0025_02', 'scene0583_00', 'scene0583_01', 'scene0583_02',
    'scene0701_00', 'scene0701_01', 'scene0701_02', 'scene0580_00',
    'scene0580_01', 'scene0565_00', 'scene0169_00', 'scene0169_01',
    'scene0655_00', 'scene0655_01', 'scene0655_02', 'scene0063_00',
    'scene0221_00', 'scene0221_01', 'scene0591_00', 'scene0591_01',
    'scene0591_02', 'scene0678_00', 'scene0678_01', 'scene0678_02',
    'scene0462_00', 'scene0427_00', 'scene0595_00', 'scene0193_00',
    'scene0193_01', 'scene0164_00', 'scene0164_01', 'scene0164_02',
    'scene0164_03', 'scene0598_00', 'scene0598_01', 'scene0598_02',
    'scene0599_00', 'scene0599_01', 'scene0599_02', 'scene0328_00',
    'scene0300_00', 'scene0300_01', 'scene0354_00', 'scene0458_00',
    'scene0458_01', 'scene0423_00', 'scene0423_01', 'scene0423_02',
    'scene0307_00', 'scene0307_01', 'scene0307_02', 'scene0606_00',
    'scene0606_01', 'scene0606_02', 'scene0432_00', 'scene0432_01',
    'scene0608_00', 'scene0608_01', 'scene0608_02', 'scene0651_00',
    'scene0651_01', 'scene0651_02', 'scene0430_00', 'scene0430_01',
    'scene0689_00', 'scene0357_00', 'scene0357_01', 'scene0574_00'
]
