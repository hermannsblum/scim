from sacred import Experiment
import torch
import tensorflow_datasets as tfds
import tensorflow as tf

from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')

from semsegcluster.data.tfds_to_torch import TFDataIterableDataset
from semsegcluster.data.augmentation import augmentation
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.sacred_utils import get_observer

ex = Experiment()
ex.observers.append(get_observer())

ex.add_config(
    batchsize=10,
)

@ex.main
def test(batchsize):

  traindata, info = tfds.load('oaisys11k',
                        split='train',
                        as_supervised=True, with_info=True)

  def data_converter_rugd(image, label):
    image = convert_img_to_float(image)
    label = tf.squeeze(tf.cast(label, tf.int64))

    label = tf.where(label == 0, tf.cast(255, tf.int64), label)
    label = tf.where(label == 1, tf.cast(0, tf.int64), label)
    label = tf.where(label == 2, tf.cast(255, tf.int64), label)
    label = tf.where(label == 3, tf.cast(1, tf.int64), label)
    label = tf.where(label == 4, tf.cast(2, tf.int64), label)
    label = tf.where(label == 5, tf.cast(3, tf.int64), label)
    label = tf.where(label == 6, tf.cast(4, tf.int64), label)
    label = tf.where(label == 7, tf.cast(5, tf.int64), label)
    label = tf.where(label == 8, tf.cast(3, tf.int64), label)
    label = tf.where(label == 9, tf.cast(3, tf.int64), label)
    label = tf.where(label == 10, tf.cast(255, tf.int64), label)
    label = tf.where(label == 11, tf.cast(6, tf.int64), label)
    label = tf.where(label == 12, tf.cast(255, tf.int64), label)
    label = tf.where(label == 13, tf.cast(7, tf.int64), label)
    label = tf.where(label == 14, tf.cast(8, tf.int64), label)
    label = tf.where(label == 15, tf.cast(9, tf.int64), label)
    label = tf.where(label == 16, tf.cast(3, tf.int64), label)
    label = tf.where(label == 17, tf.cast(3, tf.int64), label)
    label = tf.where(label == 18, tf.cast(3, tf.int64), label)
    label = tf.where(label == 19, tf.cast(2, tf.int64), label)
    label = tf.where(label == 20, tf.cast(3, tf.int64), label)
    label = tf.where(label == 21, tf.cast(10, tf.int64), label)
    label = tf.where(label == 22, tf.cast(255, tf.int64), label)
    label = tf.where(label == 23, tf.cast(255, tf.int64), label)
    label = tf.where(label == 24, tf.cast(3, tf.int64), label)
    label = tf.where(label >= 15, tf.cast(255, tf.int64), label)

    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

  traindata = TFDataIterableDataset(
      traindata.map(lambda x, y: augmentation(
          x, y, random_crop=(256, 256))).map(data_converter_rugd))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)


  num_grass = 0
  num_logs = 0

  size_grass = 0
  size_logs = 0

  for i, (images, targets) in tqdm(enumerate(train_loader)):
    for target in targets:
      if (target == 1).any():
        num_grass += 1
        size_grass += (target == 1).sum()
      if (target == 9).any():
        num_logs += 1
        size_logs += (target == 9).sum()
  print("Pictures with grass: ", num_grass)
  print("Pictures with logs: ", num_logs)
  if num_grass > 0:
    print("Average_size of grass: ", size_grass/num_grass)
  if num_logs > 0:
    print("Average size of logs: ", size_logs/num_logs)
  return


if __name__ == '__main__':
  ex.run_commandline()
