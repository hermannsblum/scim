import tensorflow as tf
import tensorflow_datasets as tfds


def resize_with_crop(image, shape, method='bilinear'):
    """
  Resizes an image while maintaining aspect ratio by cropping away parts of the image.
  """
    target_h, target_w = shape
    target_aspect = tf.cast(target_w, tf.float32) / tf.cast(
        target_h, tf.float32)
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
            offset_height=tf.cast(
                .5 * (image_h - image_w / target_aspect) - .5, tf.int32),
            offset_width=0,
            target_height=tf.cast(image_w / target_aspect, tf.int32),
            target_width=image_shape[1])

    return tf.image.resize(image, (target_h, target_w), method=method)


COCO_LABELNAMES = tfds.object_detection.coco.Coco(
    config=tfds.object_detection.coco.Coco.builder_configs['2017_panoptic']
).info.features['panoptic_objects']['label'].names


class CocoSegmentation(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image':
                tfds.features.Image(shape=(480, 640, 3)),
                'image/filename':
                tf.string,
                'image/id':
                tf.int64,
                'label':
                tfds.features.Image(shape=(480, 640, 1)),
            }),
            supervised_keys=('image', 'label'),
        )

    def _split_generators(self, dl_manager):
        return {
            split: self._generate_examples(
                tfds.load('coco/2017_panoptic', split=split))
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
                current_class = tf.cast(blob['panoptic_objects']['label'][i],
                                        tf.uint8)
                semantic = tf.where(panoptic_id == current_id, current_class,
                                    semantic)
            image = resize_with_crop(blob['image'], (480, 640),
                                     method='bilinear')
            semantic = resize_with_crop(semantic[..., tf.newaxis], (480, 640),
                                        method='nearest')
            yield int(blob['image/id'].numpy()), {
                'image': image.numpy().astype('uint8'),
                'image/filename': blob['image/filename'].numpy(),
                'image/id': blob['image/id'].numpy(),
                'label': semantic.numpy(),
            }
