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
import scipy as sp
import scipy.spatial

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

from deeplab.sampling import (get_dino, get_sampling_idx, get_dino_embeddings,
                              get_deeplab_embeddings, dino_color_normalize)

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


def measure_assigned_miou(cm):
  cm = cm[:40].astype(np.uint32)
  newcm = np.zeros((40, 40), dtype=np.uint32)
  for pred_c in range(cm.shape[1]):
    if pred_c == 39:
      assigned_class = 38
    else:
      assigned_class = np.argmax(cm[:, pred_c])
    if newcm[assigned_class, assigned_class] < cm[assigned_class, pred_c]:
      # count those existing assigned classifications as misclassifications
      newcm[:, 38] += newcm[:, assigned_class]
      # assign current cluster to this class
      newcm[:, assigned_class] = cm[:, pred_c]
    else:
      newcm[:, 38] += cm[:, pred_c]
  iou = np.diag(newcm) / (newcm.sum(0) + newcm.sum(1) - np.diag(newcm))
  return np.nanmean(iou)


@ex.capture
@memory.cache
def get_distances(subset, shard, subsample, device, pretrained_model, pred_name,
                  uncert_name, expected_feature_shape, uncertainty_threshold):
  out = get_sampling_idx(subset=subset,
                         shard=shard,
                         subsample=subsample,
                         expected_feature_shape=expected_feature_shape,
                         pretrained_model=pretrained_model)
  out.update(
      get_deeplab_embeddings(subset=subset,
                             shard=shard,
                             subsample=subsample,
                             device=device,
                             pretrained_model=pretrained_model,
                             expected_feature_shape=expected_feature_shape,
                             feature_name='classifier.2',
                             pred_name=pred_name,
                             uncert_name=uncert_name))
  out.update(
      get_dino_embeddings(subset=subset,
                          shard=shard,
                          subsample=subsample,
                          device=device,
                          pretrained_model=pretrained_model,
                          expected_feature_shape=expected_feature_shape))
  del out['sampling_idx']
  out['dino_distances'] = sp.spatial.distance.pdist(out['dino_features'],
                                                    metric='cosine')
  out['same_class_pair'] = np.zeros_like(out['dino_distances'], dtype=bool)
  del out['dino_distances']
  m = out['features'].shape[0]
  for j in range(m):
    for i in range(j):
      out['same_class_pair'][m * i + j - (
          (i + 2) *
          (i + 1)) // 2] = out['prediction'][i] == out['prediction'][j]
  out['inlier_pair'] = np.zeros_like(out['same_class_pair'], dtype=bool)
  is_inlier = out['uncertainty'] < uncertainty_threshold
  for j in range(m):
    for i in range(j):
      out['inlier_pair'][m * i + j - ((i + 2) * (i + 1)) // 2] = np.logical_and(
          is_inlier[i], is_inlier[j])
  out['testing_idx'] = np.logical_and(
      out['uncertainty'] < uncertainty_threshold, out['prediction'] != 255)
  return out


@ex.capture
def kmeans_inference(clusterer, subset, pretrained_model, device, normalize,
                     _run):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  model = get_dino(device=device)
  # make sure the directory exists
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)
  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy())
    image = dino_color_normalize(image).to(device)
    # run inference
    out = model.get_intermediate_layers(image, n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h = int(image.shape[2] / model.patch_embed.patch_size)
    w = int(image.shape[3] / model.patch_embed.patch_size)
    dino_features = out[0].reshape(h, w, out.shape[-1])
    if normalize:
      dino_features = torch.nn.functional.normalize(dino_features, dim=-1)
    dino_features = torchvision.transforms.functional.resize(
        dino_features.permute((2, 0, 1)),
        size=(image.shape[2], image.shape[3]),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    dino_features = dino_features.to('cpu').detach().numpy().transpose(
        [1, 2, 0])
    # query cluster assignment of closest sample
    cluster_pred = clusterer.predict(
        dino_features.reshape((-1, dino_features.shape[-1])))
    cluster_pred = cluster_pred.reshape((image.shape[2], image.shape[3]))
    # save output
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_dinokmeans{_run._id}.npy'),
            cluster_pred)


@ex.capture
def get_kmeans(algorithm, n_clusters, normalize):
  out = get_distances()
  clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                     random_state=2,
                                     copy_x=False,
                                     algorithm=algorithm)
  if normalize:
    out['dino_features'] = sklearn.preprocessing.normalize(out['dino_features'],
                                                           norm='l2')
  out['clustering'] = clusterer.fit_predict(out['dino_features'])
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
  miou = measure_assigned_miou(cm)
  return -1.0 * (0.0 if np.isnan(miou) else miou)


ex.add_config(
    subsample=100,
    shard=5,
    device='cuda',
    ignore_other=True,
    uncertainty_threshold=-3,
    expected_feature_shape=[60, 80],
    pred_name='pseudolabel-pred',
    uncert_name='pseudolabel-maxlogit-pp',
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
  out = get_distances()
  clusterer = sklearn.cluster.KMeans(n_clusters=result.x[0],
                                     random_state=2,
                                     copy_x=False,
                                     algorithm=result.x[1])
  if result.x[2]:
    out['dino_features'] = sklearn.preprocessing.normalize(out['dino_features'],
                                                           norm='l2')
  clusterer.fit(out['dino_features'])
  kmeans_inference(clusterer=clusterer, normalize=result.x[2])


@ex.main
def kmeans(algorithm, n_clusters, normalize):
  out = get_distances()
  clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                     random_state=2,
                                     copy_x=False,
                                     algorithm=algorithm)
  if normalize:
    out['dino_features'] = sklearn.preprocessing.normalize(out['dino_features'],
                                                           norm='l2')
  clusterer.fit(out['dino_features'])
  del out
  kmeans_inference(clusterer=clusterer, normalize=normalize)


if __name__ == '__main__':
  ex.run_commandline()
