import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from semseg_density.data.nyu_depth_v2 import TRAINING_LABEL_NAMES as NYU_LABEL_NAMES


def resize_with_crop(image, shape, method='bilinear'):
  """
  Resizes an image while maintaining aspect ratio by cropping away parts of the image.
  """
  target_h, target_w = shape
  target_aspect = tf.cast(target_w, tf.float32) / tf.cast(target_h, tf.float32)
  image_shape = tf.shape(image)
  image_h = tf.cast(image_shape[0], tf.float32)
  image_w = tf.cast(image_shape[1], tf.float32)
  input_aspect = image_w / image_h

  if input_aspect >= target_aspect:
    # image is too wide
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=0,
        offset_width=tf.cast(.5 * (image_w - target_aspect * image_h) - .5,
                             tf.int32),
        target_height=image_shape[0],
        target_width=tf.cast(target_aspect * image_h, tf.int32))
  else:
    # image is too high
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=tf.cast(.5 * (image_h - image_w / target_aspect) - .5,
                              tf.int32),
        offset_width=0,
        target_height=tf.cast(image_w / target_aspect, tf.int32),
        target_width=image_shape[1])

  return tf.image.resize(image, (target_h, target_w), method=method)


COCO_LABELNAMES = tfds.object_detection.coco.Coco(
    config=tfds.object_detection.coco.Coco.builder_configs['2017_panoptic']
).info.features['panoptic_objects']['label'].names

COCO_10_LABEL_NAMES = [
    'chair', 'cup', 'tv', 'couch', 'laptop', 'keyboard', 'book', 'mouse',
    'toaster', 'bottle'
]


class CocoSegmentationConfig(tfds.core.BuilderConfig):

  def __init__(self, nyu_classes=False, classes=None, **kwargs):
    super().__init__(version='0.1.0', **kwargs)
    self.nyu_classes = nyu_classes
    if classes is None:
      self.classes = range(40) if nyu_classes else range(len(COCO_LABELNAMES))
    else:
      self.classes = classes


class CocoSegmentation(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('0.1.0')

  BUILDER_CONFIGS = [
      CocoSegmentationConfig(name='all',
                             description='Standard 2017 dataset.',
                             nyu_classes=False),
      CocoSegmentationConfig(
          name='nyu',
          description=
          'Mapped to NYU classes, only images that have at least 2 classes.',
          nyu_classes=True),
      CocoSegmentationConfig(
          name='nyu-nobag',
          description=
          'Mapped to NYU classes, only images that have at least 2 classes but no bag.',
          nyu_classes=True,
          classes=[i for i in range(40) if not NYU_LABEL_NAMES[i] in ('bag')]),
      CocoSegmentationConfig(
          name='nyu-nobook',
          description=
          'Mapped to NYU classes, only images that have at least 2 classes but no book.',
          nyu_classes=True,
          classes=[i for i in range(40) if not NYU_LABEL_NAMES[i] in ('books')]),
      CocoSegmentationConfig(
          name='10-subset',
          description='Only images that have these 10 classes.',
          nyu_classes=False,
      ),
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(480, 640, 3)),
            'image/filename': tf.string,
            'image/id': tf.int64,
            'label': tfds.features.Image(shape=(480, 640, 1)),
        }),
        supervised_keys=('image', 'label'),
    )

  def _split_generators(self, dl_manager):
    return {
        split:
        self._generate_examples(tfds.load('coco/2017_panoptic', split=split))
        for split in ('train', 'validation')
    }

  def _generate_examples(self, coco_ds):
    for blob in coco_ds:
      panoptic = tf.cast(blob['panoptic_image'], tf.int64)
      semantic = 255 * tf.ones(tf.shape(panoptic)[:2], tf.uint8)
      # the panoptic image encodes the ID of each instance into RGB, we decode it here
      panoptic_id = panoptic[
          ..., 0] + 256 * panoptic[..., 1] + 256**2 * panoptic[..., 2]
      # now we have to match the ids to their classes
      for i in range(tf.shape(blob['panoptic_objects']['id'])[0]):
        current_id = blob['panoptic_objects']['id'][i]
        current_class = tf.cast(blob['panoptic_objects']['label'][i], tf.uint8)
        semantic = tf.where(panoptic_id == current_id, current_class, semantic)
      image = resize_with_crop(blob['image'], (480, 640), method='bilinear')
      semantic = resize_with_crop(semantic[..., tf.newaxis], (480, 640),
                                  method='nearest').numpy()
      if self.builder_config.nyu_classes:
        semantic = COCO_TO_NYU40[semantic]
        contained_labels = np.unique(semantic)
        if contained_labels.shape[0] < 3:
          # only use images that have 2 classes + ignored pixels
          continue
      elif self.builder_config.name == '10-subset':
        semantic = COCO_TO_10[semantic]
        contained_labels = np.unique(semantic)
        if contained_labels.shape[0] < 2:
          # only use images that have at least 1 class
          continue
      else:
        contained_labels = np.unique(semantic)
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
      yield int(blob['image/id'].numpy()), {
          'image': image.numpy().astype('uint8'),
          'image/filename': blob['image/filename'].numpy(),
          'image/id': blob['image/id'].numpy(),
          'label': semantic,
      }


COCO_TO_NYU40_NAME = {
    'person': 'person',
    'backpack': 'bag',
    'handbag': 'bag',
    'chair': 'chair',
    'couch': 'sofa',
    'bed': 'bed',
    'light': 'lamp',
    'dining table': 'table',
    'toilet': 'toilet',
    'tv': 'television',
    'sink': 'sink',
    'refrigerator': 'refridgerator',
    'book': 'books',
    'counter': 'counter',
    'curtain': 'curtain',
    'floor-wood': 'floor',
    'door-stuff': 'door',
    'pillow': 'pilow',
    'shelf': 'shelves',
    'towel': 'towel',
    'wall-brick': 'wall',
    'wall-stone': 'wall',
    'wall-tile': 'wall',
    'wall-wood': 'wall',
    'window-blind': 'blinds',
    'window-other': 'window',
    'ceiling-merged': 'ceiling',
    'paper-merged': 'paper',
    'floor-other-merged': 'floor',
}

COCO_TO_NYU40 = 255 * np.ones(256, dtype='uint8')
for i, coco_label_name in enumerate(COCO_LABELNAMES):
  if coco_label_name in COCO_TO_NYU40_NAME:
    for j, nyu_label_name in enumerate(NYU_LABEL_NAMES):
      if COCO_TO_NYU40_NAME[coco_label_name] == nyu_label_name:
        COCO_TO_NYU40[i] = j

COCO_TO_10 = 255 * np.ones(256, dtype='uint8')
for i, coco_label_name in enumerate(COCO_LABELNAMES):
  if coco_label_name in COCO_10_LABEL_NAMES:
    for j, labelname in enumerate(COCO_10_LABEL_NAMES):
      if coco_label_name == labelname:
        COCO_TO_10[i] = j
