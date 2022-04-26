import tensorflow as tf

from semsegcluster.data.images import convert_img_to_float

#@tf.function
def augmentation(image, label, random_crop=None):
  # make sure image is in float space
  image = convert_img_to_float(image)
  # make sure that label has a last dimension
  added_label_dim = False
  if len(tf.shape(label)) < 3:
    label = label[..., tf.newaxis]
    added_label_dim = True
  # do some augmentations on image and label together
  combined = tf.concat((image, tf.cast(label, tf.float32)), axis=-1)
  combined = tf.image.random_flip_left_right(combined)
  if random_crop is not None:
    combined = tf.image.random_crop(combined, (random_crop[0],  random_crop[1], 4))
  image = combined[...,  :3]
  label = tf.cast(combined[..., 3:], label.dtype)
   # brightness
  image = tf.image.random_brightness(image, max_delta=.1)
  # hue
  image = tf.image.random_hue(image, max_delta=.1)
  # undo label dimensions
  if added_label_dim:
    label = label[..., 0]
  return image, label
