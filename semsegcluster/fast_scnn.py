"""
Implementation from https://github.com/kshitizrimal/Fast-SCNN/blob/master/tf_2_0_fast_scnn.py
"""

import tensorflow as tf
import tensorflow_addons as tfa
"""
# Model Architecture
#### Custom function for conv2d: conv_block
"""


def conv_block(inputs,
               conv_type,
               kernel,
               kernel_size,
               strides,
               normalization_type='batch',
               padding='same',
               relu=True):
  assert (normalization_type in ['batch', 'group'])

  if (conv_type == 'ds'):
    x = tf.keras.layers.SeparableConv2D(kernel,
                                        kernel_size,
                                        padding=padding,
                                        strides=strides)(inputs)
  else:
    x = tf.keras.layers.Conv2D(kernel,
                               kernel_size,
                               padding=padding,
                               strides=strides)(inputs)

  if (normalization_type == "batch"):
    x = tf.keras.layers.BatchNormalization()(x)
  else:
    # 32 is default value for the number of groups in GN. The number of input
    # channels here is `kernel`.
    x = tfa.layers.GroupNormalization(groups=min(32, kernel))(x)

  if (relu):
    x = tf.keras.activations.relu(x)

  return x


"""## Step 1: Learning to DownSample"""

# Input Layer


def _downsampling(inputs, num_downsampling_layers, normalization_type):
  lds_layer = conv_block(inputs,
                         'conv',
                         32, (3, 3),
                         strides=(2, 2),
                         normalization_type=normalization_type)
  for _ in range(1, num_downsampling_layers):
    lds_layer = conv_block(lds_layer,
                           'ds',
                           64, (3, 3),
                           strides=(2, 2),
                           normalization_type=normalization_type)
  return lds_layer


"""## Step 2: Global Feature Extractor
#### residual custom method
"""


def _res_bottleneck(inputs, filters, kernel, t, s, normalization_type, r=False):
  assert (normalization_type in ['batch', 'group'])

  tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

  x = conv_block(inputs,
                 'conv',
                 tchannel, (1, 1),
                 strides=(1, 1),
                 normalization_type=normalization_type)

  x = tf.keras.layers.DepthwiseConv2D(kernel,
                                      strides=(s, s),
                                      depth_multiplier=1,
                                      padding='same')(x)

  if (normalization_type == "batch"):
    x = tf.keras.layers.BatchNormalization()(x)
  else:
    # 32 is default value for the number of groups in GN. The number of input
    # channels here is `tchannel` * `depth_multiplier` = `tchannel`.
    x = tfa.layers.GroupNormalization(groups=min(32, tchannel))(x)

  x = tf.keras.activations.relu(x)

  x = conv_block(x,
                 'conv',
                 filters, (1, 1),
                 strides=(1, 1),
                 normalization_type=normalization_type,
                 padding='same',
                 relu=False)

  if r:
    x = tf.keras.layers.add([x, inputs])
  return x


"""#### Bottleneck custom method"""


def bottleneck_block(inputs, filters, kernel, t, strides, n,
                     normalization_type):
  x = _res_bottleneck(inputs,
                      filters,
                      kernel,
                      t,
                      strides,
                      normalization_type=normalization_type)

  for i in range(1, n):
    x = _res_bottleneck(x,
                        filters,
                        kernel,
                        t,
                        1,
                        normalization_type=normalization_type,
                        r=True)

  return x


"""#### PPM Method"""


def pyramid_pooling_block(input_tensor, bin_sizes, h=15, w=20):
  concat_list = [input_tensor]

  for bin_size in bin_sizes:
    x = tf.keras.layers.AveragePooling2D(pool_size=(h // bin_size,
                                                    w // bin_size),
                                         strides=(
                                             h // bin_size,
                                             w // bin_size,
                                         ))(input_tensor)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (h, w)))(x)

    concat_list.append(x)

  return tf.keras.layers.concatenate(concat_list)


"""#### Assembling all the methods"""


def fast_scnn(input_shape,
              num_downsampling_layers=3,
              num_classes=19,
              normalization_type='batch'):
  assert (normalization_type in ['batch', 'group'])
  inputs = tf.keras.Input(shape=input_shape)
  inputs_converted = tf.image.convert_image_dtype(inputs, tf.float32)

  lds_layer = _downsampling(inputs_converted,
                            num_downsampling_layers=num_downsampling_layers,
                            normalization_type=normalization_type)

  gfe_layer = bottleneck_block(lds_layer,
                               64, (3, 3),
                               t=6,
                               strides=2,
                               n=3,
                               normalization_type=normalization_type)
  gfe_layer = bottleneck_block(gfe_layer,
                               96, (3, 3),
                               t=6,
                               strides=2,
                               n=3,
                               normalization_type=normalization_type)
  gfe_layer = bottleneck_block(gfe_layer,
                               128, (3, 3),
                               t=6,
                               strides=1,
                               n=3,
                               normalization_type=normalization_type)

  downsampling_factor = 4 * 2**num_downsampling_layers
  gfe_layer = pyramid_pooling_block(gfe_layer, [2, 4, 6, 8],
                                    h=input_shape[0] // downsampling_factor,
                                    w=input_shape[1] // downsampling_factor)
  """## Step 3: Feature Fusion"""

  ff_layer1 = conv_block(lds_layer,
                         'conv',
                         128, (1, 1),
                         padding='same',
                         strides=(1, 1),
                         normalization_type=normalization_type,
                         relu=False)

  ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)

  ff_layer2 = tf.keras.layers.SeparableConv2D(128, (3, 3),
                                              padding='same',
                                              strides=(1, 1),
                                              activation=None,
                                              dilation_rate=(4, 4))(ff_layer2)

  # old approach with DepthWiseConv2d
  #ff_layer2 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
  if (normalization_type == "batch"):
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
  else:
    # 32 is default value for the number of groups in GN. The number of input
    # channels here is 128.
    ff_layer2 = tfa.layers.GroupNormalization(groups=32)(ff_layer2)

  ff_layer2 = tf.keras.activations.relu(ff_layer2)
  ff_layer2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same',
                                     activation=None)(ff_layer2)

  ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
  if (normalization_type == "batch"):
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
  else:
    # 32 is default value for the number of groups in GN. The number of input
    # channels here is 128.
    ff_final = tfa.layers.GroupNormalization(groups=32)(ff_final)

  ff_final = tf.keras.activations.relu(ff_final)
  """## Step 4: Classifier"""

  classifier = tf.keras.layers.SeparableConv2D(
      128, (3, 3), padding='same', strides=(1, 1),
      name='DSConv1_classifier')(ff_final)
  if (normalization_type == "batch"):
    classifier = tf.keras.layers.BatchNormalization()(classifier)
  else:
    # 32 is default value for the number of groups in GN. The number of input
    # channels here is 128.
    classifier = tfa.layers.GroupNormalization(groups=32)(classifier)
  classifier = tf.keras.activations.relu(classifier)

  classifier = tf.keras.layers.SeparableConv2D(
      128, (3, 3), padding='same', strides=(1, 1),
      name='DSConv2_classifier')(classifier)
  if (normalization_type == "batch"):
    classifier = tf.keras.layers.BatchNormalization()(classifier)
  else:
    # 32 is default value for the number of groups in GN. The number of input
    # channels here is 128.
    classifier = tfa.layers.GroupNormalization(groups=32)(classifier)
  classifier = tf.keras.activations.relu(classifier)

  classifier = conv_block(classifier,
                          'conv',
                          num_classes, (1, 1),
                          strides=(1, 1),
                          normalization_type=normalization_type,
                          padding='same',
                          relu=False)

  #classifier = tf.keras.layers.Dropout(0.3)(classifier)

  classifier = tf.keras.layers.UpSampling2D(
      (2**num_downsampling_layers, 2**num_downsampling_layers))(classifier)
  # classifier = tf.keras.activations.softmax(classifier)

  encoder = tf.keras.Model(inputs=inputs, outputs=ff_final)
  model = tf.keras.Model(inputs=inputs, outputs=classifier)

  return encoder, model
