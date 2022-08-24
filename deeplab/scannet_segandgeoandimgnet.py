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
import markov_clustering as mc

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
                              get_geometric_features, get_sampling_idx,
                              get_resnet, get_imagenet_embeddings)

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


@ex.capture
@memory.cache
def get_distances(subset, shard, subsample, device, pretrained_model,
                  feature_name, imagenet_feature_name, interpolate_imagenet,
                  pred_name, uncert_name, expected_feature_shape,
                  uncertainty_threshold, normalize):
  out = get_sampling_idx(subset=subset,
                         shard=shard,
                         subsample=subsample,
                         expected_feature_shape=expected_feature_shape,
                         pretrained_model=pretrained_model)
  out.update(
      get_geometric_features(subset=subset,
                             shard=shard,
                             subsample=subsample,
                             pretrained_model=pretrained_model,
                             expected_feature_shape=expected_feature_shape))
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
  # removing points without geometric feature (ie no NANs)
  has_geometry = np.isfinite(out['geometric_features'].mean(1))
  del out['sampling_idx']
  for k in out:
    out[k] = out[k][has_geometry]
  if normalize:
    out['features'] = sklearn.preprocessing.normalize(out['features'])
    out['imagenet_features'] = sklearn.preprocessing.normalize(
        out['imagenet_features'])
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
  out['imagenet_distances'] = sp.spatial.distance.pdist(
      out['imagenet_features'], metric='euclidean')
  out['geometric_distances'] = sp.spatial.distance.pdist(
      out['geometric_features'], metric='euclidean')
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
  out['testing_idx'] = np.logical_and(
      out['uncertainty'] < uncertainty_threshold, out['prediction'] != 255)
  return out


def distance_preprocessing(geometric_weight, imagenet_weight):
  out = get_distances()
  print('Loaded all features', flush=True)
  # clear up memory
  for k in list(out.keys()):
    if k not in ('distances', 'geometric_distances', 'imagenet_distances',
                 'inlier_pair', 'same_class_pair'):
      del out[k]
  # now generate distance matrix
  for k in ['distances', 'geometric_distances', 'imagenet_distances']:
    # first scale such that 90% of inliers of same class are below 1
    inlier_dist = out[k][np.logical_and(out['inlier_pair'],
                                        out['same_class_pair'])]
    scale_idx = int(0.9 * inlier_dist.shape[0])
    scale_value = np.partition(inlier_dist, scale_idx)[scale_idx]
    #_run.info['scale_value'] = scale_value
    out[k] /= scale_value
  adjacency = sp.spatial.distance.squareform(
      (1 - geometric_weight - imagenet_weight) * out['distances'] +
      geometric_weight * out['geometric_distances'] +
      imagenet_weight * out['imagenet_distances'])
  out = get_distances()
  return {
      'features': out['features'],
      'geometric_features': out['geometric_features'],
      'imagenet_features': out['imagenet_features'],
      'adjacency': adjacency,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
      'testing_idx': out['testing_idx'],
  }


@ex.capture
def clustering_based_inference(features, geometric_features, imagenet_features,
                               clustering, subset, pretrained_model,
                               feature_name, imagenet_feature_name,
                               interpolate_imagenet, expected_feature_shape,
                               device, geometric_weight, imagenet_weight,
                               normalize, _run, postfix):
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
  # and for geometric features
  geo_knn = hnswlib.Index(space='l2', dim=geometric_features.shape[-1])
  geo_knn.init_index(max_elements=geometric_features.shape[0],
                     M=64,
                     ef_construction=200,
                     random_seed=100)
  geo_knn.add_items(geometric_features, clustering)
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
  # load geometric features
  with open(os.path.join(directory, 'blockid_to_descriptor.pkl'), 'rb') as f:
    blockid_to_descriptor = pickle.load(f)
  feature_voxels = np.array(list(blockid_to_descriptor.keys()))
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
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
    del imagenet_features, image
    try:
      voxel = np.load(os.path.join(
          directory,
          f'{frame}_pseudolabel-voxels.npy')).squeeze().astype(np.int32)
      # reshapes to have array <feature_width * feature_height, list of voxels>
      voxel = voxel.reshape((expected_feature_shape[0],
                             voxel.shape[0] // expected_feature_shape[0],
                             expected_feature_shape[1],
                             voxel.shape[1] // expected_feature_shape[1]))
      voxel = np.swapaxes(voxel, 1, 2).reshape(
          (expected_feature_shape[0] * expected_feature_shape[1], -1))
      has_feature = np.isin(voxel, feature_voxels)
      # out of the possible voxels, pick one that has a feature
      voxel = voxel[range(voxel.shape[0]),
                    np.argmax(has_feature, axis=-1)].flatten()
      has_feature = has_feature.max(-1).flatten()
      geometric_features = []
      for v in voxel:
        if v in blockid_to_descriptor:
          geometric_features.append(blockid_to_descriptor[v])
        else:
          geometric_features.append(np.array([np.nan for _ in range(32)]))
      del voxel
      geometric_features = np.stack(geometric_features, axis=0)
      geo_pred, geo_distance = geo_knn.knn_query(geometric_features, k=1)
      geo_distance[np.logical_not(has_feature)] = 1e10
      del has_feature, geometric_features
      in_distance = in_distance / imagenet_weight
      geo_distance = geo_distance / geometric_weight
      cluster_pred = np.where(in_distance < geo_distance, in_pred, geo_pred)
      cluster_dist = np.where(in_distance < geo_distance, in_distance,
                              geo_distance)
      del in_pred, in_distance, geo_pred, geo_distance
    except FileNotFoundError:
      cluster_pred = in_pred
      cluster_dist = in_distance
    dl_weight = max(1e-10, 1 - imagenet_weight - geometric_weight)
    cluster_pred = np.where((dl_distance / dl_weight) < cluster_dist, dl_pred,
                            cluster_pred)
    del cluster_dist, dl_pred, dl_distance
    cluster_pred = cluster_pred.reshape((feature_shape[1], feature_shape[2]))
    cluster_pred = cv2.resize(cluster_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_seggeoimg{postfix}{_run._id}.npy'),
            cluster_pred)


hdbscan_dimensions = [
    Integer(name='min_cluster_size', low=2, high=100),
    Integer(name='min_samples', low=1, high=30),
    Real(name='geometric_weight', low=0, high=1),
    Real(name='imagenet_weight', low=0, high=1),
]


@ex.capture
def get_hdbscan(cluster_selection_method, min_cluster_size, min_samples,
                geometric_weight, imagenet_weight):
  adjacency = distance_preprocessing(
      geometric_weight=geometric_weight,
      imagenet_weight=imagenet_weight)['adjacency']
  clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                              min_samples=int(min_samples),
                              cluster_selection_method=cluster_selection_method,
                              metric='precomputed')
  clustering = clusterer.fit_predict(adjacency)
  out = distance_preprocessing(geometric_weight=geometric_weight,
                               imagenet_weight=imagenet_weight)
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=hdbscan_dimensions)
def score_hdbscan(geometric_weight, imagenet_weight, min_cluster_size,
                  min_samples):
  if geometric_weight + imagenet_weight >= 1:
    return 1.
  out = get_hdbscan(geometric_weight=float(geometric_weight),
                    imagenet_weight=float(imagenet_weight),
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples))
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  measures = measure_from_confusion_matrix(cm.astype(np.uint32))
  miou = measures['assigned_miou']
  vscore = measures['v_score']
  print(f'{miou=:.3f}, {vscore=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


dbscan_dimensions = [
    Real(name='eps', low=0.1, high=1),
    Integer(name='min_samples', low=1, high=40),
    Real(name='geometric_weight', low=0, high=1),
    Real(name='imagenet_weight', low=0, high=1),
]


@ex.capture
def get_dbscan(eps, min_samples, geometric_weight, imagenet_weight):
  adjacency = distance_preprocessing(
      geometric_weight=geometric_weight,
      imagenet_weight=imagenet_weight)['adjacency']
  # TODO check where negative values come from. Numerical issue?
  adjacency[adjacency < 0] = 0
  clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                     min_samples=min_samples,
                                     metric='precomputed')
  clustering = clusterer.fit_predict(adjacency)
  out = distance_preprocessing(geometric_weight=geometric_weight,
                               imagenet_weight=imagenet_weight)
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=dbscan_dimensions)
def score_dbscan(eps, min_samples, imagenet_weight, geometric_weight):
  if float(geometric_weight) + float(imagenet_weight) >= 1:
    return 1.
  out = get_dbscan(eps=float(eps),
                   min_samples=int(min_samples),
                   imagenet_weight=float(imagenet_weight),
                   geometric_weight=float(imagenet_weight))
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  measures = measure_from_confusion_matrix(cm.astype(np.uint32))
  miou = measures['assigned_miou']
  vscore = measures['v_score']
  print(f'{miou=:.3f}, {vscore=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


mcl_dimensions = [
    Real(name='eta', low=0.1, high=20),
    Real(name='inflation', low=1.2, high=2.5),
    Real(name='geometric_weight', low=0, high=1),
    Real(name='imagenet_weight', low=0, high=1),
]


@ex.capture
def get_mcl(eta, inflation, geometric_weight, imagenet_weight):
  out = distance_preprocessing(geometric_weight=geometric_weight,
                               imagenet_weight=imagenet_weight)
  adjacency = out['adjacency']
  del out
  # add their activation function
  adjacency = np.exp(-1.0 * eta * adjacency)
  # now run the clustering
  result = mc.run_mcl(adjacency / adjacency.mean(),
                      inflation=inflation,
                      verbose=True)
  clusters = mc.get_clusters(result)
  print(f'Fit clustering to {len(clusters)} clusters', flush=True)
  out = distance_preprocessing(geometric_weight=imagenet_weight)
  clustering = -1 * np.ones(out['features'].shape[0], dtype=int)
  for i, cluster in enumerate(clusters):
    for node in cluster:
      clustering[node] = i
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=mcl_dimensions)
def score_mcl(eta, inflation, geometric_weight):
  if geometric_weight + imagenet_weight >= 1:
    return 1.
  out = get_mcl(eta=eta, inflation=inflation, geometric_weight=imagenet_weight)
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
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
  _run.info['geometric_weight'] = result.x[2]
  _run.info['imagenet_weight'] = result.x[3]
  # run clustering again with best parameters
  out = get_hdbscan(min_cluster_size=result.x[0],
                    min_samples=result.x[1],
                    geometric_weight=result.x[2],
                    imagenet_weight=result.x[3])
  clustering_based_inference(features=out['features'],
                             geometric_features=out['geometric_features'],
                             imagenet_features=out['imagenet_features'],
                             geometric_weight=result.x[2],
                             imagenet_weight=result.x[3],
                             postfix='hdbscan',
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
  _run.info['geometric_weight'] = result.x[2]
  _run.info['imagenet_weight'] = result.x[3]
  # run clustering again with best parameters
  out = get_dbscan(eps=result.x[0],
                   min_samples=result.x[1],
                   geometric_weight=result.x[2],
                   imagenet_weight=result.x[3])
  clustering_based_inference(features=out['features'],
                             geometric_features=out['geometric_features'],
                             imagenet_features=out['imagenet_features'],
                             geometric_weight=result.x[2],
                             imagenet_weight=result.x[3],
                             postfix='dbscan',
                             clustering=out['clustering'])


@ex.command
def best_mcl(
    _run,
    n_calls,
):
  # run optimisation
  result = gp_minimize(func=score_mcl,
                       dimensions=mcl_dimensions,
                       n_calls=n_calls,
                       random_state=4)
  _run.info['best_miou'] = -result.fun
  _run.info['eta'] = result.x[0]
  _run.info['inflation'] = result.x[1]
  _run.info['geometric_weight'] = result.x[2]
  # run clustering again with best parameters
  out = get_mcl(eta=result.x[0],
                inflation=result.x[1],
                geometric_weight=result.x[2])
  clustering_based_inference(features=out['features'],
                             geometric_weight=result.x[2],
                             postfix='mcl',
                             clustering=out['clustering'])


if __name__ == '__main__':
  ex.run_commandline()
