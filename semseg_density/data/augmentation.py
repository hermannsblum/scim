import tensorflow as tf

#@tf.function
def augmentation(image, label):
  # make sure image is in float space
  image = tf.image.convert_image_dtype(image, tf.float32)
  # make sure that label has a last dimension
  added_label_dim = False
  if len(tf.shape(label)) < 3:
    label = label[..., tf.newaxis]
    added_label_dim = True
  # random flip
  if tf.random.uniform((1,)) < .5:
    image = tf.image.flip_left_right(image)
    label = tf.image.flip_left_right(label)
  # brightness
  image = tf.image.random_brightness(image, max_delta=60)
  # hue
  image = tf.image.random_hue(image, max_delta=.1)
  # undo label dimensions
  if added_label_dim:
    label = label[..., 0]
  return image, label
