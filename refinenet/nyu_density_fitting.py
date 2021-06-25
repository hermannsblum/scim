from sacred import Experiment
import torch
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from semseg_density.data.tfds_to_torch import TFDataIterableDataset
from semseg_density.data.augmentation import augmentation
from semseg_density.model.refinenet import rf_lw50
from semseg_density.model.refinenet_uncertainty import RefineNetDensity
from semseg_density.gdrive import load_gdrive_file
from semseg_density.lr_scheduler import LRScheduler
from semseg_density.segmentation_metrics import SegmentationMetric
from semseg_density.losses import MixSoftmaxCrossEntropyLoss
from semseg_density.settings import TMPDIR
from semseg_density.sacred_utils import get_observer

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
          n_components=40,
          covariance_type='full',
          batchsize=8,
          reg_covar=1e-6,
          device='cuda',
          pretrained_model='adelaine'):
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
  model = rf_lw50(40, pretrained=pretrained_model == 'adelaine')
  # Load pretrained weights
  if pretrained_model and pretrained_model != 'adelaine':
    checkpoint = torch.load(load_gdrive_file(pretrained_model, ending='pth'))
    load_checkpoint(model, checkpoint, strict=False)
  model.to(device)

  # Create hook to get features from intermediate pytorch layer
  hooks = {}

  def get_activation(name, features=hooks):

    def hook(model, input, output):
      features['feat'] = output.detach()

    return hook

  # get feature layer
  feature_layer = getattr(model, "mflow_conv_g4_pool")
  # register hook to get features
  feature_layer.register_forward_hook(get_activation("mflow_conv_g4_pool"))

  start_time = time.time()
  model.eval()

  all_features = []
  for i, (images, _) in enumerate(train_loader):
    images = images.to(device)

    out = model(images)
    features = hooks['feat']
    features = features.to('cpu').detach().numpy()
    # reshaping
    features = features.transpose([0, 2, 3, 1]).reshape([-1, 256])
    # subsampling (because storing all these embeddings would be too much)
    features = features[np.random.choice(features.shape[0],
                                         size=[500],
                                         replace=False)]
    all_features.append(features)
  all_features = np.concatenate(all_features, axis=0)
  print('Loaded all features', flush=True)

  # fit PCA
  pca = PCA(64, copy=False)
  small_features = pca.fit_transform(all_features)
  print('PCA fit!', flush=True)

  # fit GMM
  gmm = GaussianMixture(
      n_components=n_components,
      covariance_type=covariance_type,
      reg_covar=reg_covar,
  )
  gmm.fit(small_features)
  del small_features
  print('GMM fit!', flush=True)

  # check the score on the validation data
  all_features = []
  for i, (images, _) in enumerate(val_loader):
    images = images.to(device)

    out = model(images)
    features = hooks['feat']
    features = features.to('cpu').detach().numpy()
    # reshaping
    features = features.transpose([0, 2, 3, 1]).reshape([-1, 256])
    # subsampling (because storing all these embeddings would be too much)
    features = features[np.random.choice(features.shape[0],
                                         size=[500],
                                         replace=False)]
    all_features.append(features)
  all_features = np.concatenate(all_features, axis=0)

  all_features = pca.transform(all_features)

  _run.info['score'] = gmm.score(all_features)
  _run.info['gmm'] = {
      'means': gmm.means_,
      'weights': gmm.weights_,
      'covariances': gmm.covariances_,
  }
  _run.info['pca'] = {
      'mean': pca.mean_,
      'components': pca.components_,
  }

  # reshape covariance matrix
  cov = gmm.covariances_
  if covariance_type == 'tied':
    # covariance for each component is the same
    cov = np.tile(np.expand_dims(cov, 0), (n_components, 1, 1))
  elif covariance_type == 'diag':
    # transform from diagonal vector to matrix
    newcov = np.zeros((n_components, cov.shape[-1], cov.shape[-1]))
    for i in range(n_components):
      # np.diag only works on 1-dimensional arrays
      newcov[i] = np.diag(cov[i])
    cov = newcov
  cov = torch.as_tensor(cov)

  densitymodel = RefineNetDensity(40,
                                  size=50,
                                  n_components=n_components,
                                  pca_mean=torch.as_tensor(pca.mean_),
                                  pca_components=torch.as_tensor(
                                      pca.components_.T),
                                  means=torch.as_tensor(gmm.means_),
                                  covariances=cov,
                                  weights=torch.as_tensor(gmm.weights_))
  filename = 'refinenet_nyu_density.pth'
  save_path = os.path.join(TMPDIR, filename)
  torch.save(densitymodel.state_dict(), save_path)
  _run.add_artifact(save_path)


if __name__ == '__main__':
  ex.run_commandline()
