from sacred import Experiment
import torch
import torchvision
import PIL
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
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

import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile
import hnswlib

import semsegcluster.data.scannet
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.segmentation_metrics import SegmentationMetric
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.eval_munkres import measure_from_confusion_matrix
from semsegcluster.sacred_utils import get_observer, get_checkpoint, get_incense_loader

from deeplab.sampling import (get_deeplab_embeddings, get_dino, get_deeplab,
                              get_geometric_features, get_sampling_idx,
                              get_dino_embeddings, dino_color_normalize)

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


@ex.capture
@memory.cache
def get_distances(subset, shard, subsample, device, pretrained_model,
                  feature_name, pred_name, uncert_name, expected_feature_shape,
                  uncertainty_threshold):
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
      get_dino_embeddings(subset=subset,
                          shard=shard,
                          subsample=subsample,
                          device=device,
                          pretrained_model=pretrained_model,
                          expected_feature_shape=expected_feature_shape))
  # removing points without geometric feature (ie no NANs)
  has_geometry = np.isfinite(out['geometric_features'].mean(1))
  del out['sampling_idx']
  for k in out:
    out[k] = out[k][has_geometry]
  for k in ['features', 'dino_features']:
    out[k] = sklearn.preprocessing.normalize(out[k])
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
  out['dino_distances'] = sp.spatial.distance.pdist(out['dino_features'],
                                                    metric='euclidean')
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
  out['distance_scaling'] = {}
  for k in ['distances', 'geometric_distances', 'dino_distances']:
    # first scale such that 90% of inliers of same class are below 1
    inlier_dist = out[k][np.logical_and(out['inlier_pair'],
                                        out['same_class_pair'])]
    scale_idx = int(0.9 * inlier_dist.shape[0])
    scale_value = np.partition(inlier_dist, scale_idx)[scale_idx]
    #_run.info['scale_value'] = scale_value
    out[k] /= scale_value
    out['distance_scaling'][k] = scale_value
  return out


def distance_preprocessing(geometric_weight, dino_weight):
  out = get_distances()
  # now generate distance matrix
  adjacency = sp.spatial.distance.squareform(
      (1 - geometric_weight - dino_weight) * out['distances'] +
      geometric_weight * out['geometric_distances'] +
      dino_weight * out['dino_distances'])
  return {
      'features': out['features'],
      'geometric_features': out['geometric_features'],
      'dino_features': out['dino_features'],
      'adjacency': adjacency,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
      'testing_idx': out['testing_idx'],
      'distance_scaling': out['distance_scaling'],
  }


@ex.capture
def clustering_based_inference(features,
                               geometric_features,
                               dino_features,
                               distance_scaling,
                               clustering,
                               subset,
                               pretrained_model,
                               feature_name,
                               expected_feature_shape,
                               device,
                               geometric_weight,
                               dino_weight,
                               normalize,
                               _run,
                               postfix,
                               save=True):
  # set up NN index to find closest clustered sample
  dl_knn = hnswlib.Index(space='l2', dim=features.shape[-1])
  dl_knn.init_index(max_elements=features.shape[0],
                    M=64,
                    ef_construction=200,
                    random_seed=100)
  dl_knn.add_items(features, clustering)
  # onother KNN for dino features
  in_knn = hnswlib.Index(space='l2', dim=dino_features.shape[-1])
  in_knn.init_index(max_elements=dino_features.shape[0],
                    M=64,
                    ef_construction=200,
                    random_seed=100)
  in_knn.add_items(dino_features, clustering)
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
  dino_model = get_dino(device)
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
    image = torch.from_numpy(image.numpy())
    dino_image = dino_color_normalize(image).to(device)
    image = image.to(device)
    # run inference
    _ = dl_model(image)['out']
    features = dl_hooks['feat']
    image = image.cpu()
    del image
    features = torch.nn.functional.normalize(features, dim=1)
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    feature_shape = features.shape
    assert features.shape[-1] == 256
    # query cluster assignment of closest sample
    dl_pred, dl_distance = dl_knn.knn_query(features.reshape((-1, 256)), k=1)
    dl_distance /= distance_scaling['distances']
    del features
    dino_features = dino_model.get_intermediate_layers(dino_image, n=1)[0]
    dino_features = dino_features[:, 1:, :]  # we discard the [CLS] token
    h = int(dino_image.shape[2] / dino_model.patch_embed.patch_size)
    w = int(dino_image.shape[3] / dino_model.patch_embed.patch_size)
    dino_features = dino_features[0].reshape(h, w, dino_features.shape[-1])
    dino_features = torch.nn.functional.normalize(dino_features, dim=-1)
    dino_features = torchvision.transforms.functional.resize(
        dino_features.permute((2, 0, 1)),
        size=expected_feature_shape,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    dino_features = dino_features.to('cpu').detach().numpy().transpose(
        [1, 2, 0])
    in_pred, in_distance = in_knn.knn_query(dino_features.reshape(
        (-1, dino_features.shape[-1])),
                                            k=1)
    in_distance /= distance_scaling['dino_distances']
    dino_image = dino_image.cpu()
    del dino_features, dino_image
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
      geo_distance /= distance_scaling['geometric_distances']
      geo_distance[np.logical_not(has_feature)] = 1e10
      del has_feature, geometric_features
      in_distance = in_distance / dino_weight
      geo_distance = geo_distance / geometric_weight
      cluster_pred = np.where(in_distance < geo_distance, in_pred, geo_pred)
      cluster_dist = np.where(in_distance < geo_distance, in_distance,
                              geo_distance)
      del in_pred, in_distance, geo_pred, geo_distance
    except FileNotFoundError:
      cluster_pred = in_pred
      cluster_dist = in_distance
    dl_weight = max(1e-10, 1 - dino_weight - geometric_weight)
    cluster_pred = np.where((dl_distance / dl_weight) < cluster_dist, dl_pred,
                            cluster_pred)
    del cluster_dist, dl_pred, dl_distance
    cluster_pred = cluster_pred.reshape((feature_shape[1], feature_shape[2]))
    cluster_pred = cv2.resize(cluster_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    name = blob['name'].numpy().decode()
    if save:
      # save output
      np.save(
          os.path.join(directory, f'{name}_seggeodino{postfix}{_run._id}.npy'),
          cluster_pred)
    yield name, cluster_pred


hdbscan_dimensions = [
    Integer(name='min_cluster_size', low=2, high=100),
    Integer(name='min_samples', low=1, high=5),
    Real(name='geometric_weight', low=0, high=1),
    Real(name='dino_weight', low=0, high=1),
]


@ex.capture
def get_hdbscan(cluster_selection_method, min_cluster_size, min_samples,
                geometric_weight, dino_weight):
  out = distance_preprocessing(geometric_weight=geometric_weight,
                               dino_weight=dino_weight)
  adjacency = out['adjacency']
  clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                              min_samples=int(min_samples),
                              cluster_selection_method=cluster_selection_method,
                              metric='precomputed')
  clustering = clusterer.fit_predict(adjacency)
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=hdbscan_dimensions)
def score_hdbscan(geometric_weight, dino_weight, min_cluster_size, min_samples):
  if float(geometric_weight) + float(dino_weight) >= 1:
    return 1.
  out = get_hdbscan(geometric_weight=float(geometric_weight),
                    dino_weight=float(dino_weight),
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples))
  # make space for wildcard cluster 39
  out['clustering'][out['clustering'] >= 39] += 1
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  miou = measure_from_confusion_matrix(cm)['assigned_miou']
  return -1.0 * (0.0 if np.isnan(miou) else miou)


dbscan_dimensions = [
    Real(name='eps', low=0.1, high=1),
    Integer(name='min_samples', low=1, high=40),
    Real(name='geometric_weight', low=0, high=1),
    Real(name='dino_weight', low=0, high=1),
]


@ex.capture
def get_dbscan(eps, min_samples, geometric_weight, dino_weight):
  out = distance_preprocessing(geometric_weight=geometric_weight,
                               dino_weight=dino_weight)
  adjacency = out['adjacency']
  # TODO check where negative values come from. Numerical issue?
  adjacency[adjacency < 0] = 0
  clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                     min_samples=min_samples,
                                     metric='precomputed')
  clustering = clusterer.fit_predict(adjacency)
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=dbscan_dimensions)
def score_dbscan(eps, min_samples, dino_weight, geometric_weight):
  if float(geometric_weight) + float(dino_weight) >= 1:
    return 1.
  out = get_dbscan(eps=float(eps),
                   min_samples=int(min_samples),
                   dino_weight=float(dino_weight),
                   geometric_weight=float(dino_weight))
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  miou = measure_from_confusion_matrix(cm)['assigned_miou']
  return -1.0 * (0.0 if np.isnan(miou) else miou)


mcl_dimensions = [
    Real(name='eta', low=0.1, high=20),
    Real(name='inflation', low=1.2, high=2.5),
    Real(name='geometric_weight', low=0, high=1),
    Real(name='dino_weight', low=0, high=1),
]


@ex.capture
def get_mcl(eta, inflation, geometric_weight, dino_weight):
  out = distance_preprocessing(geometric_weight=geometric_weight,
                               dino_weight=dino_weight)
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
  out = distance_preprocessing(geometric_weight=dino_weight)
  clustering = -1 * np.ones(out['features'].shape[0], dtype=int)
  for i, cluster in enumerate(clusters):
    for node in cluster:
      clustering[node] = i
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=mcl_dimensions)
def score_mcl(eta, inflation, geometric_weight):
  if geometric_weight + dino_weight >= 1:
    return 1.
  out = get_mcl(eta=eta, inflation=inflation, geometric_weight=dino_weight)
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
              dino_feature_name='layer4',
              interpolate_dino=True,
              expected_feature_shape=[60, 80],
              ignore_other=True,
              uncertainty_threshold=-3,
              dino_feature='block.5',
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
  _run.info['min_cluster_size'] = int(result.x[0])
  _run.info['min_samples'] = int(result.x[1])
  _run.info['geometric_weight'] = result.x[2]
  _run.info['dino_weight'] = result.x[3]
  _run.info['optimisation_outcomes'] = list(result['func_vals'])
  _run.info['optimisation_points'] = result['x_iters']
  # run clustering again with best parameters
  out = get_hdbscan(min_cluster_size=result.x[0],
                    min_samples=result.x[1],
                    geometric_weight=result.x[2],
                    dino_weight=result.x[3])
  for _ in clustering_based_inference(
      features=out['features'],
      geometric_features=out['geometric_features'],
      distance_scaling=out['distance_scaling'],
      dino_features=out['dino_features'],
      geometric_weight=result.x[2],
      dino_weight=result.x[3],
      postfix='hdbscan',
      save=True,
      clustering=out['clustering']):
    continue


@ex.command
def run_hdbscan(
    _run,
    min_samples,
    min_cluster_size,
    geometric_weight,
    dino_weight,
):
  out = get_hdbscan(min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    geometric_weight=geometric_weight,
                    dino_weight=dino_weight)
  for _ in clustering_based_inference(
      features=out['features'],
      geometric_features=out['geometric_features'],
      distance_scaling=out['distance_scaling'],
      dino_features=out['dino_features'],
      geometric_weight=geometric_weight,
      dino_weight=dino_weight,
      postfix='hdbscan',
      save=True,
      clustering=out['clustering']):
    continue


@ex.command
def evaluate_hdbscan_optimisation(_run, optimisation, subset, pretrained_model):
  loader = get_incense_loader()
  run = loader.find_by_id(optimisation)
  assert run.config['subset'] == subset
  assert run.config['pretrained_model'] == pretrained_model
  _, pretrained_id = get_checkpoint(run.config['pretrained_model'])
  directory = os.path.join(EXP_OUT, 'scannet_inference', run.config['subset'],
                           pretrained_id)
  outcomes = []
  mious = []
  for i, p in enumerate(run.info['optimisation_points']):
    if i % 5 != 0:
      continue
    out = get_hdbscan(
        cluster_selection_method=run.config['cluster_selection_method'],
        min_cluster_size=p[0],
        min_samples=p[1],
        geometric_weight=p[2],
        dino_weight=p[3])
    cm = torchmetrics.ConfusionMatrix(num_classes=200)
    for frame, pred in clustering_based_inference(
        features=out['features'],
        geometric_features=out['geometric_features'],
        distance_scaling=out['distance_scaling'],
        dino_features=out['dino_features'],
        geometric_weight=p[2],
        dino_weight=p[3],
        postfix='None',
        save=False,
        clustering=out['clustering']):
      label = np.load(os.path.join(directory, f'{frame}_label.npy'))
      label[label >= 37] = 255
      # update confusion matrix, only on labelled pixels
      if np.any(label != 255):
        label = torch.from_numpy(label)
        # make space for wildcard cluster 39
        pred[pred >= 39] += 1
        # handle nans as misclassification
        pred[pred == -1] = 39
        pred[np.isnan(pred)] = 39
        # cluster numbers larger than 200 are ignored in  the confusionm  matrix
        pred[pred > 200] = 39
        pred = torch.from_numpy(pred)[label != 255]
        label = label[label != 255]
        cm.update(pred, label)
    cm = cm.compute().numpy().astype(np.uint32)
    outcomes.append(run.info['optimisation_outcomes'][i])
    miou = measure_from_confusion_matrix(cm)['assigned_miou']
    mious.append(miou)
  _run.info['optimisation_outcomes'] = outcomes
  _run.info['mious'] = mious


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
  _run.info['dino_weight'] = result.x[3]
  # run clustering again with best parameters
  out = get_dbscan(eps=result.x[0],
                   min_samples=result.x[1],
                   geometric_weight=result.x[2],
                   dino_weight=result.x[3])
  clustering_based_inference(features=out['features'],
                             geometric_features=out['geometric_features'],
                             dino_features=out['dino_features'],
                             geometric_weight=result.x[2],
                             dino_weight=result.x[3],
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
