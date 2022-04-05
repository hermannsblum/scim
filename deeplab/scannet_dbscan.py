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

import semseg_density.data.scannet
from semseg_density.data.tfds_to_torch import TFDataIterableDataset
from semseg_density.data.augmentation import augmentation
from semseg_density.data.images import convert_img_to_float
from semseg_density.gdrive import load_gdrive_file
from semseg_density.segmentation_metrics import SegmentationMetric
from semseg_density.settings import TMPDIR, EXP_OUT
from semseg_density.sacred_utils import get_observer, get_checkpoint
from semseg_density.eval import measure_from_confusion_matrix

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
  all_voxels = np.array([], dtype=np.int32)
  all_labels = []
  all_uncertainties = []
  all_entropies = []
  for blob in tqdm(data.shard(shard, 0)):
    frame = blob['name'].numpy().decode()
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
    assert features.shape[-1] == 256
    label = np.load(os.path.join(directory,
                                 f'{frame}_{pred_name}.npy')).squeeze()
    uncert = np.load(os.path.join(directory,
                                  f'{frame}_{uncert_name}.npy')).squeeze()
    voxel = np.load(os.path.join(
        directory,
        f'{frame}_pseudolabel-voxels.npy')).squeeze().astype(np.int32)
    # interpolate to feature size
    label = cv2.resize(label,
                       dsize=(features.shape[2], features.shape[1]),
                       interpolation=cv2.INTER_NEAREST)
    assert label.shape[0] == features.shape[1]
    assert label.shape[1] == features.shape[2]
    label = label.flatten()
    # reshapes to have array <feature_width * feature_height, list of voxels>
    voxel = voxel.reshape(
        (features.shape[1], voxel.shape[0] // features.shape[1],
         features.shape[2], voxel.shape[1] // features.shape[2]))
    voxel = np.swapaxes(voxel, 1, 2).reshape(
        (features.shape[1] * features.shape[2], -1))
    uncert = cv2.resize(uncert,
                        dsize=(features.shape[2], features.shape[1]),
                        interpolation=cv2.INTER_LINEAR).flatten()
    entropy = torchvision.transforms.functional.resize(
        entropy,
        size=(features.shape[1], features.shape[2]),
        interpolation=PIL.Image.BILINEAR).to('cpu').detach().numpy().flatten()
    features = features.reshape((-1, 256))
    # subsampling (because storing all these embeddings would be too much)
    # first subsample from voxels that we have already sampled
    already_sampled = np.isin(voxel, all_voxels)
    assert len(already_sampled.shape) == 2
    already_sampled_idx = already_sampled.max(-1)
    if already_sampled_idx.sum() > 0:
      # sample 20% from known voxels
      sampled_idx = np.flatnonzero(already_sampled_idx)
      if sampled_idx.shape[0] > subsample // 5:
        # reduce to a subset
        sampled_idx = np.random.choice(sampled_idx,
                                       size=[subsample // 5],
                                       replace=False)
    else:
      sampled_idx = np.array([], dtype=np.int32)
    # now add random samples
    sampled_idx = np.concatenate(
        (sampled_idx,
         np.random.choice(features.shape[0],
                          size=[subsample - sampled_idx.shape[0]],
                          replace=False)),
        axis=0)
    all_features.append(features[sampled_idx])
    all_labels.append(label[sampled_idx])
    all_uncertainties.append(uncert[sampled_idx])
    all_entropies.append(entropy[sampled_idx])
    # if a feature corresponds to multiple voxels, favor those that are already sampled
    already_sampled = already_sampled[sampled_idx]
    all_voxels = np.concatenate(
        (all_voxels, voxel[sampled_idx, np.argmax(already_sampled, axis=1)]), axis=0)
    del logits, image, features, label, voxel, uncert, already_sampled

  return {
      'features': np.concatenate(all_features, axis=0),
      'voxels': all_voxels,
      'prediction': np.concatenate(all_labels, axis=0),
      'uncertainty': np.concatenate(all_uncertainties, axis=0),
      'entropy': np.concatenate(all_entropies, axis=0),
  }


@ex.capture
@memory.cache
def get_distances(subset, shard, subsample, device, pretrained_model,
                  feature_name, pred_name, uncert_name, uncertainty_threshold,
                  normalize):
  out = get_embeddings(subset=subset,
                       shard=shard,
                       subsample=subsample,
                       device=device,
                       pretrained_model=pretrained_model,
                       feature_name=feature_name,
                       pred_name=pred_name,
                       uncert_name=uncert_name)
  if normalize:
    out['features'] = sklearn.preprocessing.normalize(out['features'])
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
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
    for i in range(j):
      out['same_voxel_pair'][m * i + j - (
          (i + 2) * (i + 1)) // 2] = out['voxels'][i] == out['voxels'][j]
  return out


@ex.capture
def distance_preprocessing(apply_scaling, same_voxel_close,
                           distance_activation_factor, _run):
  out = get_distances()
  print('Loaded all features', flush=True)
  # now generate distance matrix
  if apply_scaling:
    # first scale such that 90% of inliers of same class are below 1
    inlier_dist = out['distances'][np.logical_and(out['inlier_pair'],
                                                  out['same_class_pair'])]
    scale_idx = int(0.9 * inlier_dist.shape[0])
    scale_value = np.partition(inlier_dist, scale_idx)[scale_idx]
    #_run.info['scale_value'] = scale_value
    out['distances'] /= scale_value
  if same_voxel_close is not None:
    # now set distances between same voxel observations to 0.5
    out['distances'][out['same_voxel_pair']] = float(
        same_voxel_close) * out['distances'][out['same_voxel_pair']]
  if distance_activation_factor is not None:
    out['distances'] = np.exp(distance_activation_factor *
                              (out['distances'] - 1))
  adjacency = sp.spatial.distance.squareform(out['distances'])
  return {
      'features': out['features'],
      'adjacency': adjacency,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
  }


@ex.capture
def clustering_based_inference(features, clustering, subset, pretrained_model,
                               device, normalize, _run):
  # set up NN index to find closest clustered sample
  knn = hnswlib.Index(space='l2', dim=256)
  knn.init_index(max_elements=features.shape[0],
                 M=64,
                 ef_construction=200,
                 random_seed=100)
  knn.add_items(features, clustering)
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
    cluster_pred, _ = knn.knn_query(features.reshape((-1, 256)), k=1)
    cluster_pred = cluster_pred.reshape((feature_shape[1], feature_shape[2]))
    cluster_pred = cv2.resize(cluster_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_hdbscan{_run._id}.npy'),
            cluster_pred)


hdbscan_dimensions = [
    Categorical(name='same_voxel_close',
                categories=[None, .1, .2, .3, .4, .5, .6, .7, .8, .9]),
    Integer(name='min_cluster_size', low=2, high=100),
    Integer(name='min_samples', low=1, high=30),
]


@ex.capture
def get_hdbscan(apply_scaling, same_voxel_close, distance_activation_factor,
                cluster_selection_method, min_cluster_size, min_samples):
  out = distance_preprocessing(apply_scaling, same_voxel_close,
                               distance_activation_factor)
  clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                              min_samples=int(min_samples),
                              cluster_selection_method=cluster_selection_method,
                              metric='precomputed')
  clustering = clusterer.fit_predict(out['adjacency'])
  return {
      'features': out['features'],
      'clustering': clustering,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
  }


@use_named_args(dimensions=hdbscan_dimensions)
def score_hdbscan(same_voxel_close, min_cluster_size, min_samples):
  out = get_hdbscan(same_voxel_close=same_voxel_close,
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples))
  out['clustering'][out['clustering'] == -1] = 39
  cm = sklearn.metrics.confusion_matrix(
      out['prediction'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      out['clustering'][np.logical_and(out['uncertainty'] < -3,
                                       out['prediction'] != 255)],
      labels=list(range(400)))
  miou = measure_from_confusion_matrix(cm.astype(np.uint32))['assigned_miou']
  print(f'{miou=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


dbscan_dimensions = [
    Real(name='eps', low=0.2, high=10),
    Integer(name='min_samples', low=1, high=40),
]


@ex.capture
def get_dbscan(apply_scaling, same_voxel_close, distance_activation_factor, eps,
               min_samples):
  out = distance_preprocessing(apply_scaling, same_voxel_close,
                               distance_activation_factor)
  clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                     min_samples=min_samples,
                                     metric='precomputed')
  clustering = clusterer.fit_predict(out['adjacency'])
  return {
      'features': out['features'],
      'clustering': clustering,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
  }


@use_named_args(dimensions=dbscan_dimensions)
def score_dbscan(eps, min_samples):
  out = get_dbscan(eps=eps, min_samples=min_samples)
  out['clustering'][out['clustering'] == -1] = 39
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
    device='cuda',
    feature_name='classifier.2',
    ignore_other=True,
    apply_scaling=True,
    same_voxel_close=None,
    use_hdbscan=False,
    eps=2,
    min_samples=3,
    uncertainty_threshold=-3,
    min_cluster_size=15,
    distance_activation_factor=None,
    normalize=True,
)


@ex.command
def best_hdbscan(
    _run,
    ignore_other,
    pretrained_model,
    device,
    subset,
    n_calls,
):
  # run optimisation
  result = gp_minimize(func=score_hdbscan,
                       dimensions=hdbscan_dimensions,
                       n_calls=n_calls,
                       random_state=4)
  _run.info['best_miou'] = -result.fun
  _run.info['same_voxel_close'] = result.x[0]
  _run.info['min_cluster_size'] = result.x[1]
  _run.info['min_samples'] = result.x[2]
  # run clustering again with best parameters
  out = get_hdbscan(same_voxel_close=result.x[0],
                    min_cluster_size=result.x[1],
                    min_samples=result.x[2])
  clustering_based_inference(features=out['features'],
                             clustering=out['clustering'])


@ex.command
def best_dbscan(
    _run,
    ignore_other,
    pretrained_model,
    device,
    subset,
    n_calls,
):
  # run optimisation
  result = gp_minimize(func=score_dbscan,
                       dimensions=dbscan_dimensions,
                       n_calls=n_calls,
                       random_state=4)
  _run.info['best_miou'] = result.fun
  _run.info['eps'] = result.x[0]
  _run.info['min_samples'] = result.x[1]
  # run clustering again with best parameters
  out = get_dbscan(eps=result.x[0], min_samples=result.x[1])
  clustering_based_inference(features=out['features'],
                             clustering=out['clustering'])


@ex.main
def dbscan(
    _run,
    ignore_other,
    min_cluster_size,
    pretrained_model,
    device,
    eps,
    min_samples,
    use_hdbscan,
    apply_scaling,
    same_voxel_close,
    distance_activation_factor,
    subset,
):
  if use_hdbscan:
    out = get_hdbscan(apply_scaling, same_voxel_close,
                      distance_activation_factor, min_cluster_size)
  else:
    out = get_dbscan(apply_scaling, same_voxel_close,
                     distance_activation_factor, eps, min_samples)
  cm = sklearn.metrics.confusion_matrix(out['prediction'], out['clustering'])
  _run.info['clustering_measurements'] = measure_from_confusion_matrix(cm)
  clustering_based_inference(features=out['features'],
                             clustering=out['clustering'])


if __name__ == '__main__':
  ex.run_commandline()
