from sacred import Experiment
import torch
import torchvision
import PIL
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
import scipy as sp
import scipy.spatial
import pickle
from tqdm import tqdm
import cv2
from collections import defaultdict
from joblib import Memory
import sklearn
import sklearn.cluster
import sklearn.preprocessing
import sklearn.metrics
import hdbscan
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
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.segmentation_metrics import SegmentationMetric
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.eval import measure_from_confusion_matrix
from semsegcluster.sacred_utils import get_observer, get_checkpoint

from deeplab.sampling import (get_deeplab_embeddings, get_deeplab,
                              get_imagenet_embeddings, get_sampling_idx,
                              get_resnet)

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


@ex.capture
@memory.cache
def get_distances(subset, shard, subsample, device, pretrained_model,
                  feature_name, imagenet_feature_name, pred_name, uncert_name,
                  expected_feature_shape, uncertainty_threshold, normalize,
                  interpolate_imagenet):
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
                             feature_name=feature_name,
                             pred_name=pred_name,
                             uncert_name=uncert_name))
  out.update(
      get_imagenet_embeddings(subset=subset,
                              shard=shard,
                              subsample=subsample,
                              device=device,
                              pretrained_model=pretrained_model,
                              expected_feature_shape=expected_feature_shape,
                              interpolate=interpolate_imagenet,
                              feature_name=imagenet_feature_name))
  if normalize:
    out['features'] = sklearn.preprocessing.normalize(out['features'])
    out['imagenet_features'] = sklearn.preprocessing.normalize(
        out['imagenet_features'])
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
  out['imagenet_distances'] = sp.spatial.distance.pdist(
      out['imagenet_features'], metric='euclidean')
  out['same_class_pair'] = np.zeros_like(out['distances'], dtype=bool)
  m = out['features'].shape[0]
  for j in range(m):
    for i in range(j):
      out['same_class_pair'][m * i + j - (
          (i + 2) *
          (i + 1)) // 2] = out['prediction'][i] == out['prediction'][j]
  out['inlier_pair'] = np.zeros_like(out['distances'], dtype=bool)
  is_inlier = out['uncertainty'] < uncertainty_threshold
  for j in range(m):
    for i in range(j):
      out['inlier_pair'][m * i + j - ((i + 2) * (i + 1)) // 2] = np.logical_and(
          is_inlier[i], is_inlier[j])
  del is_inlier
  out['same_voxel_pair'] = np.zeros_like(out['distances'], dtype=bool)
  for j in range(m):
    if out['voxels'][j] == 0:
      continue
    for i in range(j):
      if out['voxels'][i] == 0:
        continue
      out['same_voxel_pair'][m * i + j - (
          (i + 2) * (i + 1)) // 2] = out['voxels'][i] == out['voxels'][j]
  return out


def distance_preprocessing(imagenet_weight, same_voxel_close):
  out = get_distances()
  print('Loaded all features', flush=True)
  # clear up memory
  del out['features']
  del out['imagenet_features']
  del out['voxels']
  # now generate distance matrix
  for k in ['distances', 'imagenet_distances']:
    # first scale such that 90% of inliers of same class are below 1
    inlier_dist = out[k][np.logical_and(out['inlier_pair'],
                                        out['same_class_pair'])]
    scale_idx = int(0.9 * inlier_dist.shape[0])
    scale_value = np.partition(inlier_dist, scale_idx)[scale_idx]
    #_run.info['scale_value'] = scale_value
    out[k] /= scale_value
  merged = ((1 - imagenet_weight) * out['distances'] +
            imagenet_weight * out['imagenet_distances'])
  if same_voxel_close is not None:
    # now set distances between same voxel observations to 0.5
    merged[out['same_voxel_pair']] = float(same_voxel_close) * merged[
        out['same_voxel_pair']]
  adjacency = sp.spatial.distance.squareform(merged)
  out = get_distances()
  return {
      'features': out['features'],
      'imagenet_features': out['imagenet_features'],
      'adjacency': adjacency,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
  }


@ex.capture
def clustering_based_inference(features, imagenet_features, clustering, subset,
                               pretrained_model, feature_name,
                               expected_feature_shape, imagenet_feature_name,
                               device, imagenet_weight, interpolate_imagenet,
                               normalize, _run):
  # set up NN index to find closest clustered sample
  dl_knn = hnswlib.Index(space='l2', dim=features.shape[-1])
  dl_knn.init_index(max_elements=features.shape[0],
                    M=64,
                    ef_construction=200,
                    random_seed=100)
  dl_knn.add_items(features, clustering)
  # onother KNN for imagenet features
  in_knn = hnswlib.Index(space='l2', dim=imagenet_features.shape[-1])
  in_knn.init_index(max_elements=imagenet_features.shape[0],
                    M=64,
                    ef_construction=200,
                    random_seed=100)
  in_knn.add_items(imagenet_features, clustering)
  data = tfds.load(f'scan_net/{subset}', split='validation')
  dl_model, dl_hooks = get_deeplab(pretrained_model=pretrained_model,
                                   feature_name=feature_name,
                                   device=device)
  in_model, in_hooks = get_resnet(feature_name=imagenet_feature_name,
                                  device=device)
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
    _ = dl_model(image)['out']
    features = dl_hooks['feat']
    if normalize:
      features = torch.nn.functional.normalize(features, dim=1)
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    feature_shape = features.shape
    assert features.shape[-1] == 256
    # query cluster assignment of closest sample
    dl_pred, dl_distance = dl_knn.knn_query(features.reshape((-1, 256)), k=1)
    del features
    _ = in_model(image)
    imagenet_features = in_hooks['feat']
    if normalize:
      imagenet_features = torch.nn.functional.normalize(imagenet_features,
                                                        dim=1)
    if interpolate_imagenet:
      imagenet_features = torchvision.transforms.functional.resize(
          imagenet_features,
          size=expected_feature_shape,
          interpolation=PIL.Image.BILINEAR)
    imagenet_features = imagenet_features.to('cpu').detach().numpy().transpose(
        [0, 2, 3, 1])[0]
    in_pred, in_distance = in_knn.knn_query(imagenet_features.reshape(
        (-1, imagenet_features.shape[-1])),
                                            k=1)
    del imagenet_features
    cluster_pred = np.where((dl_distance / (1 - imagenet_weight)) <
                            (in_distance / imagenet_weight), dl_pred, in_pred)
    cluster_pred = cluster_pred.reshape((feature_shape[1], feature_shape[2]))
    cluster_pred = cv2.resize(cluster_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_segimgnetvoxel{_run._id}.npy'),
            cluster_pred)


hdbscan_dimensions = [
    Integer(name='min_cluster_size', low=2, high=100),
    Integer(name='min_samples', low=1, high=30),
    Real(name='imagenet_weight', low=0, high=1),
    Real(name='same_voxel_close', low=0, high=1),
]


@ex.capture
def get_hdbscan(cluster_selection_method, min_cluster_size, min_samples,
                imagenet_weight, same_voxel_close):
  out = distance_preprocessing(imagenet_weight=imagenet_weight,
                               same_voxel_close=same_voxel_close)
  clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                              min_samples=int(min_samples),
                              cluster_selection_method=cluster_selection_method,
                              metric='precomputed')
  clustering = clusterer.fit_predict(out['adjacency'])
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=hdbscan_dimensions)
def score_hdbscan(imagenet_weight, min_cluster_size, min_samples,
                  same_voxel_close):
  out = get_hdbscan(imagenet_weight=float(imagenet_weight),
                    same_voxel_close=float(same_voxel_close),
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples))
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(
      out['prediction'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      out['clustering'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      labels=list(range(400)))
  measures = measure_from_confusion_matrix(cm.astype(np.uint32))
  miou = measures['assigned_miou']
  vscore = measures['v_score']
  print(f'{miou=:.3f}, {vscore=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


dbscan_dimensions = [
    Real(name='eps', low=0.1, high=3),
    Integer(name='min_samples', low=1, high=40),
    Real(name='imagenet_weight', low=0, high=1),
    Real(name='same_voxel_close', low=0, high=1),
]


@ex.capture
def get_dbscan(eps, min_samples, imagenet_weight, same_voxel_close):
  out = distance_preprocessing(imagenet_weight=imagenet_weight,
                               same_voxel_close=same_voxel_close)
  adjacency = out['adjacency']
  # del out
  clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                     min_samples=min_samples,
                                     metric='precomputed')
  clustering = clusterer.fit_predict(adjacency)
  # out = distance_preprocessing(imagenet_weight=imagenet_weight)
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=dbscan_dimensions)
def score_dbscan(eps, min_samples, imagenet_weight, same_voxel_close):
  out = get_dbscan(eps=eps,
                   min_samples=min_samples,
                   same_voxel_close=same_voxel_close,
                   imagenet_weight=imagenet_weight)
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(
      out['prediction'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      out['clustering'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      labels=list(range(200)))
  measures = measure_from_confusion_matrix(cm.astype(np.uint32))
  miou = measures['assigned_miou']
  vscore = measures['v_score']
  print(f'{miou=:.3f}, {vscore=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


ex.add_config(subsample=100,
              shard=5,
              device='cuda',
              feature_name='classifier.2',
              imagenet_feature_name='layer4',
              interpolate_imagenet=True,
              expected_feature_shape=[60, 80],
              ignore_other=True,
              use_hdbscan=False,
              uncertainty_threshold=-3,
              normalize=True,
              pred_name='pseudolabel-pred',
              uncert_name='pseudolabel-maxlogit-pp')


@ex.command
def best_hdbscan(
    _run,
    n_calls,
):
  # run optimisation
  result = gp_minimize(func=score_hdbscan,
                       dimensions=hdbscan_dimensions,
                       n_calls=n_calls,
                       random_state=4)
  _run.info['best_miou'] = -result.fun
  _run.info['min_cluster_size'] = result.x[0]
  _run.info['min_samples'] = result.x[1]
  _run.info['imagenet_weight'] = result.x[2]
  _run.info['same_voxel_close'] = result.x[3]
  # run clustering again with best parameters
  out = get_hdbscan(min_cluster_size=result.x[0],
                    min_samples=result.x[1],
                    imagenet_weight=result.x[2],
                    same_voxel_close=result.x[3])
  clustering_based_inference(features=out['features'],
                             imagenet_features=out['imagenet_features'],
                             imagenet_weight=result.x[2],
                             clustering=out['clustering'])


@ex.command
def best_dbscan(
    _run,
    n_calls,
):
  # run optimisation
  result = gp_minimize(func=score_dbscan,
                       dimensions=dbscan_dimensions,
                       n_calls=n_calls,
                       random_state=4)
  _run.info['best_miou'] = -result.fun
  _run.info['eps'] = result.x[0]
  _run.info['min_samples'] = result.x[1]
  _run.info['imagenet_weight'] = result.x[2]
  _run.info['same_voxel_close'] = result.x[3]
  # run clustering again with best parameters
  out = get_dbscan(eps=result.x[0],
                   min_samples=result.x[1],
                   imagenet_weight=result.x[2],
                   same_voxel_close=result.x[3])
  clustering_based_inference(features=out['features'],
                             imagenet_features=out['imagenet_features'],
                             imagenet_weight=result.x[2],
                             clustering=out['clustering'])


if __name__ == '__main__':
  ex.run_commandline()
