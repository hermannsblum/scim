from sacred import Experiment
import torch
import torchvision
import PIL
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.spatial
import pickle
from tqdm import tqdm
import cv2
from collections import defaultdict
from joblib import Memory
import markov_clustering as mc
import sklearn
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

from deeplab.sampling import (get_geometric_features, get_deeplab_embeddings,
                              get_sampling_idx, get_deeplab)

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


@ex.capture
@memory.cache
def get_nakajima_distances(subset, shard, subsample, device, pretrained_model,
                           feature_name, pred_name, uncert_name,
                           expected_feature_shape, uncertainty_threshold):
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
      get_geometric_features(subset=subset,
                             shard=shard,
                             subsample=subsample,
                             pretrained_model=pretrained_model,
                             expected_feature_shape=expected_feature_shape))
  # removing points without geometric feature (ie no NANs)
  has_geometry = np.isfinite(out['geometric_features'].mean(1))
  del out['sampling_idx']
  for k in out:
    out[k] = out[k][has_geometry]
  # weight features by entropy
  entropy_weight = np.expand_dims(out.pop('entropy') / np.log(40), 1)
  out['features'] = (1 - entropy_weight) * out['features']
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
  out['geometric_features'] = entropy_weight * out['geometric_features']
  out['geometric_distances'] = sp.spatial.distance.pdist(
      out['geometric_features'], metric='euclidean')
  out['same_class_pair'] = np.zeros_like(out['distances'], dtype=bool)
  m = out['features'].shape[0]
  for j in range(m):
    if out['prediction'][j] == 255:
      continue
    for i in range(j):
      if out['prediction'][i] == 255:
        continue
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
  out['testing_idx'] = np.logical_and(
      out['uncertainty'] < uncertainty_threshold, out['prediction'] != 255)
  return out


@ex.capture
def nakajima_inference(weighted_features, geometric_features, clustering,
                       subset, pretrained_model, device, feature_name, _run):
  # set up NN index to find closest clustered sample
  knn = hnswlib.Index(space='l2', dim=256)
  knn.init_index(max_elements=weighted_features.shape[0],
                 M=64,
                 ef_construction=200,
                 random_seed=100)
  knn.add_items(weighted_features, clustering)
  # onother KNN for imagenet features
  geo_knn = hnswlib.Index(space='l2', dim=geometric_features.shape[-1])
  geo_knn.init_index(max_elements=geometric_features.shape[0],
                     M=64,
                     ef_construction=200,
                     random_seed=100)
  geo_knn.add_items(geometric_features, clustering)
  data = tfds.load(f'scan_net/{subset}', split='validation')
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
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
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)
    # run inference
    logits = model(image)['out']
    entropy = torch.distributions.categorical.Categorical(
        logits=logits.permute(0, 2, 3, 1)).entropy()
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    feature_shape = features.shape
    assert features.shape[-1] == 256
    # interpolate entropy to feature size
    entropy = torchvision.transforms.functional.resize(
        entropy,
        size=(features.shape[1], features.shape[2]),
        interpolation=PIL.Image.BILINEAR).to('cpu').detach().numpy()
    features = features.reshape((-1, 256))
    entropy = entropy.flatten()
    # weight features by entropy:
    features = np.expand_dims(1 - entropy / np.log(40), 1) * features
    # query cluster assignment of closest sample
    dl_pred, dl_distance = knn.knn_query(features, k=1)
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
      # weight features by entropy:
      geometric_features = np.expand_dims(entropy / np.log(40),
                                          1) * geometric_features
      geo_pred, geo_distance = geo_knn.knn_query(geometric_features, k=1)
      del geometric_features
      geo_distance[np.logical_not(has_feature)] = 1e10
      cluster_pred = np.where(dl_distance < geo_distance, dl_pred, geo_pred)
    except FileNotFoundError:
      cluster_pred = dl_pred
    # rescale to output size
    cluster_pred = cluster_pred.reshape((feature_shape[1], feature_shape[2]))
    cluster_pred = cv2.resize(cluster_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_nakajima{_run._id}.npy'),
            cluster_pred)


mcl_nakajima_dimensions = [
    Real(name='eta', low=0.1, high=20),
    Real(name='inflation', low=1.2, high=2.5),
]


@ex.capture
def get_mcl_nakajima(eta, inflation):
  out = get_nakajima_distances()
  n_points = out['features'].shape[0]
  distances = out['distances'] + out['geometric_distances']
  del out
  # add their activation function
  distances = np.exp(-1.0 * eta * distances)
  # put into square form
  adjacency = sp.spatial.distance.squareform(distances)
  # now run the clustering
  result = mc.run_mcl(adjacency / adjacency.mean(),
                      inflation=inflation,
                      verbose=True)
  clusters = mc.get_clusters(result)
  print(f'Fit clustering to {len(clusters)} clusters', flush=True)
  clustering = -1 * np.ones(n_points, dtype=int)
  for i, cluster in enumerate(clusters):
    for node in cluster:
      clustering[node] = i
  out = get_nakajima_distances()
  return {
      'features': out['features'],
      'geometric_features': out['geometric_features'],
      'testing_idx': out['testing_idx'],
      'clustering': clustering,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
  }


@use_named_args(dimensions=mcl_nakajima_dimensions)
def score_mcl_nakajima(eta, inflation):
  out = get_mcl_nakajima(eta=eta, inflation=inflation)
  out['clustering'][out['clustering'] == -1] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  miou = measure_from_confusion_matrix(cm.astype(np.uint32))['assigned_miou']
  print(f'{miou=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


ex.add_config(subsample=100,
              shard=5,
              device='cuda',
              uncertainty_threshold=-3,
              pred_name='pseudolabel-pred',
              uncert_name='pseudolabel-maxlogit-pp',
              expected_feature_shape=[60, 80],
              feature_name='classifier.2',
              ignore_other=True)


@ex.command
def best_mcl_nakajima(n_calls, _run):
  # run optimisation
  result = gp_minimize(func=score_mcl_nakajima,
                       dimensions=mcl_nakajima_dimensions,
                       n_calls=n_calls,
                       random_state=4)
  _run.info['best_miou'] = result.fun
  _run.info['best_eta'] = result.x[0]
  _run.info['best_inflation'] = result.x[1]
  # run clustering with best parameters
  out = get_mcl_nakajima(eta=result.x[0], inflation=result.x[1])
  nakajima_inference(weighted_features=out['features'],
                     geometric_features=out['geometric_features'],
                     clustering=out['clustering'])


@ex.command
def mcl_nakajima():
  out = get_mcl_nakajima()
  nakajima_inference(weighted_features=out['features'],
                     clustering=out['clustering'])


if __name__ == '__main__':
  ex.run_commandline()
