from sacred import Experiment
import torch
import torchvision
import PIL
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import cv2
from collections import defaultdict
from joblib import Memory
import sklearn
import sklearn.cluster
import sklearn.preprocessing
import sklearn.metrics
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
from skopt.utils import use_named_args

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile
import hnswlib

import semsegcluster.data.scannet
from semsegcluster.data.tfds_to_torch import TFDataIterableDataset
from semsegcluster.data.augmentation import augmentation
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.gdrive import load_gdrive_file
from semsegcluster.segmentation_metrics import SegmentationMetric
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_observer, get_checkpoint
from semsegcluster.eval import measure_from_confusion_matrix

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


def load_checkpoint(model, state_dict, strict=True):
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


@ex.capture
def get_model(pretrained_model, feature_name, device):
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=40,
      aux_loss=False)
  checkpoint, pretrained_id = get_checkpoint(pretrained_model)
  # remove any aux classifier stuff
  removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
  for k in removekeys:
    del checkpoint[k]
  load_checkpoint(model, checkpoint)
  model.to(device)
  model.eval()

  # Create hook to get features from intermediate pytorch layer
  hooks = {}

  def get_activation(name, features=hooks):

    def hook(model, input, output):
      features['feat'] = output.detach()

    return hook

  # register hook to get features
  for n, m in model.named_modules():
    if n == feature_name:
      m.register_forward_hook(get_activation(feature_name))
  return model, hooks


@ex.capture
@memory.cache
def get_embeddings(subset, shard, subsample, device, pretrained_model,
                   feature_name, pred_name, uncert_name):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  model, hooks = get_model(pretrained_model=pretrained_model,
                           feature_name=feature_name,
                           device=device)
  all_features = []
  all_labels = []
  all_uncertainties = []
  for blob in tqdm(data.shard(shard, 0)):
    frame = blob['name'].numpy().decode()
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)
    # run inference
    logits = model(image)['out']
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    label = np.load(os.path.join(directory,
                                 f'{frame}_{pred_name}.npy')).squeeze()
    uncert = np.load(os.path.join(directory,
                                  f'{frame}_{uncert_name}.npy')).squeeze()
    # interpolate to feature size
    label = cv2.resize(label,
                       dsize=(features.shape[2], features.shape[1]),
                       interpolation=cv2.INTER_NEAREST)
    assert label.shape[0] == features.shape[1]
    assert label.shape[1] == features.shape[2]
    uncert = cv2.resize(uncert,
                        dsize=(features.shape[2], features.shape[1]),
                        interpolation=cv2.INTER_LINEAR)
    features = features.reshape((-1, 256))
    label = label.flatten()
    uncert = uncert.flatten()
    # subsampling (because storing all these embeddings would be too much)
    sampled_idx = np.random.choice(features.shape[0],
                                   size=[subsample],
                                   replace=False)
    all_features.append(features[sampled_idx])
    all_labels.append(label[sampled_idx])
    all_uncertainties.append(uncert[sampled_idx])

    del logits, image, features, label,  uncert
  return {
      'features': np.concatenate(all_features, axis=0),
      'prediction': np.concatenate(all_labels, axis=0),
      'uncertainty': np.concatenate(all_uncertainties, axis=0),
  }


@ex.capture
def kmeans_inference(clusterer, subset, pretrained_model, device, normalize,
                     _run):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  model, hooks = get_model()
  # make sure the directory exists
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)
  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)
    # run inference
    logits = model(image)['out']
    features = hooks['feat']
    if normalize:
      features = torch.nn.functional.normalize(features, dim=1)
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    feature_shape = features.shape
    assert features.shape[-1] == 256
    # query cluster assignment of closest sample
    cluster_pred = clusterer.predict(features.reshape((-1, 256)))
    cluster_pred = cluster_pred.reshape((feature_shape[1], feature_shape[2]))
    cluster_pred = cv2.resize(cluster_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_kmeans{_run._id}.npy'),
            cluster_pred)


@ex.capture
def get_kmeans(algorithm, n_clusters, normalize):
  out = get_embeddings()
  clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                     random_state=2,
                                     copy_x=False,
                                     algorithm=algorithm)
  if normalize:
    out['features'] = sklearn.preprocessing.normalize(out['features'],
                                                      norm='l2')
  out['clustering'] = clusterer.fit_predict(out['features'])
  return out


kmeans_dimensions = [
    Integer(name='n_clusters', low=2, high=60),
    Categorical(name='algorithm', categories=['full', 'elkan']),
    Categorical(name='normalize', categories=[True, False]),
]


@use_named_args(dimensions=kmeans_dimensions)
def score_kmeans(n_clusters, algorithm, normalize):
  out = get_kmeans(n_clusters=n_clusters,
                   algorithm=algorithm,
                   normalize=normalize)
  cm = sklearn.metrics.confusion_matrix(
      out['prediction'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      out['clustering'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      labels=list(range(200)))
  miou = measure_from_confusion_matrix(cm.astype(np.uint32))['assigned_miou']
  print(f'{miou=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


ex.add_config(
    subsample=100,
    shard=5,
    algorithm='full',
    n_clusters=60,
    device='cuda',
    feature_name='classifier.2',
    ignore_other=True,
)


@ex.command
def best_kmeans(_run, n_calls):
  # run optimisation
  result = gp_minimize(func=score_kmeans,
                       dimensions=kmeans_dimensions,
                       n_calls=n_calls,
                       random_state=4)
  _run.info['best_miou'] = -result.fun
  _run.info['n_clusters'] = result.x[0]
  _run.info['algorithm'] = result.x[1]
  _run.info['normalize'] = result.x[2]
  # run clustering again with best parameters
  out = get_embeddings()
  clusterer = sklearn.cluster.KMeans(n_clusters=result.x[0],
                                     random_state=2,
                                     copy_x=False,
                                     algorithm=result.x[1])
  if result.x[2]:
    out['features'] = sklearn.preprocessing.normalize(out['features'],
                                                      norm='l2')
  clusterer.fit(out['features'])
  del out
  kmeans_inference(clusterer=clusterer, normalize=result.x[2])


@ex.main
def kmeans(algorithm, n_clusters, normalize):
  out = get_embeddings()
  clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                     random_state=2,
                                     copy_x=False,
                                     algorithm=algorithm)
  if normalize:
    out['features'] = sklearn.preprocessing.normalize(out['features'],
                                                      norm='l2')
  clusterer.fit(out['features'])
  del out
  kmeans_inference(clusterer=clusterer)


if __name__ == '__main__':
  ex.run_commandline()
