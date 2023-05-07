from sacred import Experiment
import torch
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
import scipy as sp
from tqdm import tqdm
import cv2
from joblib import Memory
import sklearn
import sklearn.cluster
import sklearn.preprocessing
import sklearn.metrics
import hdbscan
from skopt.callbacks import TimerCallback, EarlyStopper
from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt.utils import use_named_args

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile
import hnswlib

import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.eval_munkres import measure_from_confusion_matrix
from semsegcluster.sacred_utils import get_observer, get_checkpoint

from deeplab.oaisys_sampling import get_deeplab_embeddings, get_deeplab, get_sampling_idx

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


class DeltaYStopper(EarlyStopper):
  """Stop the optimization if the `n_best` minima are within `delta`
    Stop the optimizer if the absolute difference between the `n_best`
    objective values is less than `delta`.
    """

  def __init__(self, delta, n_best=5, n_minimum=10):
    super(EarlyStopper, self).__init__()
    self.delta = delta
    self.n_best = n_best
    self.n_minimum = n_minimum

  def _criterion(self, result):
    if len(result.func_vals) < self.n_minimum:
      return None
    if len(result.func_vals) >= self.n_best:
      func_vals = np.sort(result.func_vals)
      worst = func_vals[self.n_best - 1]
      best = func_vals[0]

      # worst is always larger, so no need for abs()
      return worst - best < self.delta

    else:
      return None


@memory.cache
def get_distances(subset, shard, subsample, device, pretrained_model,
                  feature_name, pred_name, uncert_name, uncertainty_threshold,
                  expected_feature_shape, normalize, use_euler, num_classes):
  out = get_sampling_idx(subset=subset,
                         shard=shard,
                         subsample=subsample,
                         expected_feature_shape=expected_feature_shape,
                         pretrained_model=pretrained_model,
                         use_euler=use_euler,
                         use_mapping=False)
  out.update(
      get_deeplab_embeddings(subset=subset,
                             shard=shard,
                             subsample=subsample,
                             device=device,
                             pretrained_model=pretrained_model,
                             expected_feature_shape=expected_feature_shape,
                             feature_name=feature_name,
                             pred_name=pred_name,
                             uncert_name=uncert_name,
                             use_euler=use_euler,
                             use_mapping=False,
                             num_classes=num_classes))
  if normalize:
    out['features'] = sklearn.preprocessing.normalize(out['features'])
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
  out['same_class_pair'] = np.zeros_like(out['distances'], dtype=bool)
  m = out['features'].shape[0]
  for j in tqdm(range(m)):
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
  for j in tqdm(range(m)):
    for i in range(j):
      out['inlier_pair'][m * i + j - ((i + 2) * (i + 1)) // 2] = np.logical_and(
          is_inlier[i], is_inlier[j])
  del is_inlier
  out['testing_idx'] = np.logical_and(
      out['uncertainty'] < uncertainty_threshold, out['prediction'] != 255)
  return out


@ex.capture
def distance_preprocessing(subset, shard, subsample, device, pretrained_model,
                           feature_name, pred_name, uncert_name,
                           uncertainty_threshold, expected_feature_shape,
                           normalize, apply_scaling, same_voxel_close,
                           distance_activation_factor, use_euler, num_classes, _run):
  out = get_distances(subset=subset,
                      shard=shard,
                      subsample=subsample,
                      device=device,
                      pretrained_model=pretrained_model,
                      feature_name=feature_name,
                      pred_name=pred_name,
                      uncert_name=uncert_name,
                      uncertainty_threshold=uncertainty_threshold,
                      expected_feature_shape=expected_feature_shape,
                      normalize=normalize,
                      use_euler=use_euler,
                      num_classes=num_classes)
  # now generate distance matrix
  if apply_scaling:
    # first scale such that 90% of inliers of same class are below 1
    inlier_dist = out['distances'][np.logical_and(out['inlier_pair'],
                                                  out['same_class_pair'])]
    if inlier_dist.shape[0]:
      scale_idx = int(0.9 * inlier_dist.shape[0])
      scale_value = np.partition(inlier_dist, scale_idx)[scale_idx]
      out['distances'] /= scale_value
    else:
      print("no inliers detected")
  if distance_activation_factor is not None:
    out['distances'] = np.exp(distance_activation_factor *
                              (out['distances'] - 1))
  adjacency = sp.spatial.distance.squareform(out['distances'])
  return {
      'features': out['features'],
      'adjacency': adjacency,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
      'testing_idx': out['testing_idx']
  }


@ex.capture
def clustering_based_inference(features, clustering, subset, pretrained_model,
                               feature_name, device, normalize, use_euler, num_classes, _run):
  # set up NN index to find closest clustered sample
  knn = hnswlib.Index(space='l2', dim=256)
  knn.init_index(max_elements=features.shape[0],
                 M=64,
                 ef_construction=200,
                 random_seed=100)
  knn.add_items(features, clustering)
  if use_euler:
    data = tfds.load(f'{subset}', split='validation', data_dir=f'{TMPDIR}/datasets')
  else:
    data = tfds.load(f'{subset}', split='validation')
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
                             device=device,
                             num_classes=num_classes)
  # make sure the directory exists
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)

  for idx, blob in tqdm(enumerate(data)):
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
    np.save(os.path.join(directory, f'{idx:06d}_seg{_run._id}.npy'),
            cluster_pred)


hdbscan_dimensions = [
    Real(name='same_voxel_close', low=0.1, high=1.0),
    Integer(name='min_cluster_size', low=2, high=100),
    Integer(name='min_samples', low=1, high=30),
]


@ex.capture
def get_hdbscan(apply_scaling, same_voxel_close, distance_activation_factor,
                cluster_selection_method, min_cluster_size, min_samples):
  out = distance_preprocessing(
      apply_scaling=apply_scaling,
      same_voxel_close=same_voxel_close,
      distance_activation_factor=distance_activation_factor)
  clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                              min_samples=int(min_samples),
                              cluster_selection_method=cluster_selection_method,
                              metric='precomputed')
  out['clustering'] = clusterer.fit_predict(out['adjacency'])
  return out


@use_named_args(dimensions=hdbscan_dimensions)
def score_hdbscan(same_voxel_close, min_cluster_size, min_samples):
  out = get_hdbscan(same_voxel_close=same_voxel_close,
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples))
  out['clustering'][out['clustering'] == -1] = 14
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 14
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  miou = measure_from_confusion_matrix(cm.astype(np.uint32))['assigned_miou']
  if miou > 0.99:
    print("setting mIoU it to 0.")
    miou -= 1
  print(f'{miou=:.3f}')
  return -1.0 * (0.0 if np.isnan(miou) else miou)


dbscan_dimensions = [
    Real(name='same_voxel_close', low=0.1, high=1.0),
    Real(name='eps', low=0.1, high=3),
    Integer(name='min_samples', low=1, high=40),
]


ex.add_config(
    subsample=100,
    shard=5,
    device='cuda',
    feature_name='classifier.2',
    expected_feature_shape=[60, 80],
    pred_name='pred',
    uncert_name='maxlogit-pp',
    ignore_other=True,
    apply_scaling=True,
    same_voxel_close=None,
    uncertainty_threshold=-3,
    distance_activation_factor=None,
    normalize=True,
    use_euler=False,
    subset='oaisys_trajectory',
    num_classes=15,
)


@ex.main
def best_hdbscan(
    _run,
    ignore_other,
    pretrained_model,
    device,
    subset,
    n_calls,
    use_euler,
    feature_name,
    shard,
    subsample,
    expected_feature_shape,
    pred_name,
    uncert_name,
):
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{subset}.tar')
  # run optimisation
  timer = TimerCallback()
  result = gp_minimize(func=score_hdbscan,
                       dimensions=hdbscan_dimensions,
                       callback=[
                           DeltaYStopper(0.01),
                           timer,
                       ],
                       n_calls=n_calls,
                       random_state=4)
  print(
      f"Optimisation took {np.mean(timer.iter_time):.3f}s on average, total {len(timer.iter_time)} iters."
  )
  print(f'best_miou = {-result.fun}')
  print(f'same_voxel_close = {result.x[0]}')
  print(f'min_cluster_size = {result.x[1]}')
  print(f'min_samples = {result.x[2]}')
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



if __name__ == '__main__':
  ex.run_commandline()
