from collections import OrderedDict
import tensorflow as tf

from semsegcluster.data.images import convert_img_to_float

OAISYS_LABELS = [
    'Dirt', 'Grass', 'Tree', 'Object', 'Water', 'Sky', 'Gravel', 'Mulch', 'Bedrock', 'Log', 'Rock', 'Empty_1', 'Empty_2', 'Empty_4', 'Empty_5', 'Sand'
]
OAISYS_LABELS_SHORT = [
    'Dirt', 'Grass', 'Tree', 'Object', 'Water', 'Sky', 'Gravel', 'Mulch', 'Bedrock', 'Log', 'Rock', 'Sand'
]

def data_converter_rugd(image, label):
  image = convert_img_to_float(image)
  label = tf.squeeze(tf.cast(label, tf.int64))

  label = tf.where(label == 0, tf.cast(255, tf.int64), label)
  label = tf.where(label == 1, tf.cast(0, tf.int64), label)
  label = tf.where(label == 3, tf.cast(1, tf.int64), label)
  label = tf.where(label == 9, tf.cast(3, tf.int64), label)
  label = tf.where(label == 15, tf.cast(9, tf.int64), label)
  label = tf.where(label == 2, tf.cast(15, tf.int64), label)
  label = tf.where(label == 4, tf.cast(2, tf.int64), label)
  label = tf.where(label == 5, tf.cast(3, tf.int64), label)
  label = tf.where(label == 6, tf.cast(4, tf.int64), label)
  label = tf.where(label == 7, tf.cast(5, tf.int64), label)
  label = tf.where(label == 8, tf.cast(3, tf.int64), label)
  label = tf.where(label == 10, tf.cast(255, tf.int64), label)
  label = tf.where(label == 11, tf.cast(6, tf.int64), label)
  label = tf.where(label == 12, tf.cast(255, tf.int64), label)
  label = tf.where(label == 13, tf.cast(7, tf.int64), label)
  label = tf.where(label == 14, tf.cast(8, tf.int64), label)
  label = tf.where(label == 16, tf.cast(3, tf.int64), label)
  label = tf.where(label == 17, tf.cast(3, tf.int64), label)
  label = tf.where(label == 18, tf.cast(3, tf.int64), label)
  label = tf.where(label == 19, tf.cast(2, tf.int64), label)
  label = tf.where(label == 20, tf.cast(3, tf.int64), label)
  label = tf.where(label == 21, tf.cast(10, tf.int64), label)
  label = tf.where(label == 22, tf.cast(255, tf.int64), label)
  label = tf.where(label == 23, tf.cast(255, tf.int64), label)
  label = tf.where(label == 24, tf.cast(3, tf.int64), label)
  label = tf.where(label == 30, tf.cast(15, tf.int64), label)
  label = tf.where(label > 15, tf.cast(1, tf.int64), label)  # grass label where labeled wrong by oaisys postprocessing

  # move channel from last to 2nd
  image = tf.transpose(image, perm=[2, 0, 1])
  return image, label

def load_checkpoint(model, state_dict, strict=True):
  """Load Checkpoint from Google Drive."""
  # if we currently don't use DataParallel, we have to remove the 'module' prefix
  # from all weight keys
  if (not next(iter(model.state_dict())).startswith('module')) and (next(
      iter(state_dict)).startswith('module')):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict, strict=strict)
  else:
    model.load_state_dict(state_dict, strict=strict)