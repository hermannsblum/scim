import tensorflow as tf


@tf.function
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


@tf.function
def convert_img_to_float(image):
  """
  Makes sure that the output image is in dtype float32 and has values in [0, 1[.
  There are two checks performed concerning the value rescaling:
  a) the image is currently in integer type
  b) the image has any value >= 1
  """
  needs_value_rescaling = False
  if image.dtype.is_integer:
    needs_value_rescaling = True
  image = tf.cast(image, tf.float32)
  if not needs_value_rescaling and tf.math.reduce_any(image >= 1.0):
    needs_value_rescaling = True

  if needs_value_rescaling:
    image = image / 256.0
  return image
