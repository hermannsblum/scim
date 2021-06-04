from sacred import Experiment
import torch
import tensorflow_datasets as tfds
import tensorflow as tf
import sklearn

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile

import fastscnn.data.coco_segmentation
from fastscnn.data.tfds_to_torch import TFDataIterableDataset
from fastscnn.data.augmentation import augmentation
from fastscnn.model import FastSCNN
from fastscnn.gdrive import load_gdrive_file
from fastscnn.lr_scheduler import LRScheduler
from fastscnn.segmentation_metrics import SegmentationMetric
from fastscnn.losses import MixSoftmaxCrossEntropyLoss
from fastscnn.settings import TMPDIR
from fastscnn.sacred_utils import get_observer

ex = Experiment()
ex.observers.append(get_observer())


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


@ex.main
def train(_run,
          n_components=3,
          covariance_type='tied',
          batchsize=8,
          reg_covar=1e-6,
          device='cuda',
          pretrained_model='15vSz0eBSjN4K1b3DvKZ1PQTtW4cY-_2W'):
  # DATA LOADING
  data = tfds.load('nyu_depth_v2_labeled/labeled',
                   split='train',
                   as_supervised=True)
  valdata = data.take(200)
  traindata = data.skip(200)

  def data_converter(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.int64)
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

  traindata = TFDataIterableDataset(
      traindata.cache().prefetch(10000).map(augmentation).map(data_converter))
  valdata = TFDataIterableDataset(valdata.map(data_converter))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)
  val_loader = torch.utils.data.DataLoader(dataset=valdata,
                                           batch_size=batchsize,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  model = FastSCNN(40)
  # Load pretrained weights from coco
  checkpoint = torch.load(load_gdrive_file(pretrained_model, ending='pth'))
  load_checkpoint(model, checkpoint, strict=True)
  model.to(device)

  start_time = time.time()
  model.eval()

  all_features = []
  for i, (images, _) in enumerate(train_loader):
    images = images.to(device)

    features = model(images)[-1]
    features = features.to('cpu').detach().numpy()
    # reshaping
    features = features.transpose([0, 2, 3, 1]).reshape([-1, 128])
    # subsampling (because storing all these embeddings would be too much)
    features = features[np.random.choice(features.shape[0],
                                         shape=[500],
                                         replace=False)]
    all_features.append(features)
  all_features = np.array(features)
  print('Loaded all features', flush=True)

  # fit GMM
  gmm = sklearn.mixture.GaussianMixture(
      n_components=n_components,
      covariance_type=covariance_type,
      reg_covar=reg_covar,
  )
  gmm.fit(all_features)
  del all_features
  print('GMM fit!', flush=True)

  # check the score on the validation data
  all_features = []
  for i, (images, _) in enumerate(val_loader):
    images = images.to(device)

    _, features = model(images)
    features = features.to('cpu').detach().numpy()
    # reshaping
    features = features.transpose([0, 2, 3, 1]).reshape([-1, 128])
    # subsampling (because storing all these embeddings would be too much)
    features = features[np.random.choice(features.shape[0],
                                         shape=[500],
                                         replace=False)]
    all_features.append(features)
  all_features = np.array(features)

  _run.info['score'] = gmm.score(all_features)
  _run.info['gmm'] = {
      'means': gmm.means_,
      'weights': gmm.weights_,
      'covariances': gmm.covariances_,
  }
  densitymodel = FastSCNNDensity(40,
                                 n_components=n_components,
                                 means=gmm.means_,
                                 covariances=gmm.covariances_,
                                 weights=gmm.weights_)
  filename = 'fastscnn_nyu_density.pth'
  save_path = os.path.join(TMPDIR, filename)
  torch.save(model.state_dict(), save_path)
  _run.add_artifact(save_path)


if __name__ == '__main__':
  ex.run_commandline()
