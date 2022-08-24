from sacred import Experiment
import torch
import torchvision
import torchmetrics
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
from skopt.callbacks import TimerCallback, EarlyStopper
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
from semsegcluster.sacred_utils import get_observer, get_checkpoint, get_incense_loader

from deeplab.sampling import get_deeplab_embeddings, get_deeplab, get_sampling_idx

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
                  expected_feature_shape, normalize):
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
  if normalize:
    out['features'] = sklearn.preprocessing.normalize(out['features'])
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
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
  # first scale such that 90% of inliers of same class are below 1
  inlier_dist = out['distances'][np.logical_and(out['inlier_pair'],
                                                out['same_class_pair'])]
  scale_idx = int(0.9 * inlier_dist.shape[0])
  scale_value = np.partition(inlier_dist, scale_idx)[scale_idx]
  #_run.info['scale_value'] = scale_value
  out['distances'] /= scale_value
  return out


@ex.capture
def distance_preprocessing(subset, shard, subsample, device, pretrained_model,
                           feature_name, pred_name, uncert_name,
                           uncertainty_threshold, expected_feature_shape,
                           normalize, _run):
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
                      normalize=normalize)
  # now generate distance matrix
  adjacency = sp.spatial.distance.squareform(out['distances'])
  return {
      'features': out['features'],
      'adjacency': adjacency,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
      'testing_idx': out['testing_idx']
  }


@ex.capture
def clustering_based_inference(features,
                               clustering,
                               subset,
                               pretrained_model,
                               postfix,
                               feature_name,
                               device,
                               normalize,
                               _run,
                               save=True):
  # set up NN index to find closest clustered sample
  knn = hnswlib.Index(space='l2', dim=256)
  knn.init_index(max_elements=features.shape[0],
                 M=64,
                 ef_construction=200,
                 random_seed=100)
  knn.add_items(features, clustering)
  data = tfds.load(f'scan_net/{subset}', split='validation')
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
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
    if save:
      np.save(os.path.join(directory, f'{name}_{postfix}{_run._id}.npy'),
              cluster_pred)
    else:
      yield name, cluster_pred


hdbscan_dimensions = [
    Integer(name='min_cluster_size', low=2, high=100),
    Integer(name='min_samples', low=1, high=30),
]


@ex.capture
def get_hdbscan(cluster_selection_method, min_cluster_size, min_samples):
  out = distance_preprocessing()
  clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),
                              min_samples=int(min_samples),
                              cluster_selection_method=cluster_selection_method,
                              metric='precomputed')
  clustering = clusterer.fit_predict(out['adjacency'])
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=hdbscan_dimensions)
def score_hdbscan(min_cluster_size, min_samples):
  out = get_hdbscan(min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples))
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  miou = measure_assigned_miou(cm)
  return -1.0 * (0.0 if np.isnan(miou) else miou)


dbscan_dimensions = [
    Real(name='eps', low=0.1, high=3),
    Integer(name='min_samples', low=1, high=40),
]


@ex.capture
def get_dbscan(apply_scaling, eps, min_samples):
  out = distance_preprocessing()
  clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                     min_samples=min_samples,
                                     metric='precomputed')
  clustering = clusterer.fit_predict(out['adjacency'])
  out['clustering'] = clustering
  return out


@use_named_args(dimensions=dbscan_dimensions)
def score_dbscan(eps, min_samples):
  out = get_dbscan(eps=eps, min_samples=min_samples)
  out['clustering'][out['clustering'] == -1] = 39
  # cluster numbers larger than 200 are ignored in the confusion  matrix
  out['clustering'][out['clustering'] > 200] = 39
  cm = sklearn.metrics.confusion_matrix(out['prediction'][out['testing_idx']],
                                        out['clustering'][out['testing_idx']],
                                        labels=list(range(200)))
  miou = measure_assigned_miou(cm)
  return -1.0 * (0.0 if np.isnan(miou) else miou)


ex.add_config(
    subsample=100,
    shard=5,
    device='cuda',
    feature_name='classifier.2',
    expected_feature_shape=[60, 80],
    pred_name='pseudolabel-pred',
    uncert_name='pseudolabel-maxlogit-pp',
    ignore_other=True,
    uncertainty_threshold=-3,
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
  timer = TimerCallback()
  result = gp_minimize(func=score_hdbscan,
                       dimensions=hdbscan_dimensions,
                       callback=[
                           timer,
                       ],
                       n_calls=n_calls,
                       random_state=4)
  print(
      f"Optimisation took {np.mean(timer.iter_time):.3f}s on average, total {len(timer.iter_time)} iters."
  )
  _run.info['best_miou'] = -result.fun
  _run.info['min_cluster_size'] = result.x[0]
  _run.info['min_samples'] = result.x[1]
  _run.info['optimisation_outcomes'] = list(result['func_vals'])
  _run.info['optimisation_points'] = result['x_iters']
  # run clustering again with best parameters
  out = get_hdbscan(min_cluster_size=result.x[0], min_samples=result.x[1])
  clustering_based_inference(features=out['features'],
                             clustering=out['clustering'],
                             postfix='hdbscan')


@ex.command
def best_dbscan(_run, n_calls):
  # run optimisation
  timer = TimerCallback()
  result = gp_minimize(func=score_dbscan,
                       dimensions=dbscan_dimensions,
                       callback=[
                           timer,
                       ],
                       n_calls=n_calls,
                       n_initial_points=30 if n_calls > 50 else 10,
                       random_state=4)
  print(
      f"Optimisation took {np.mean(timer.iter_time):.3f}s on average, total {len(timer.iter_time)} iters."
  )
  _run.info['best_miou'] = result.fun
  _run.info['eps'] = result.x[0]
  _run.info['min_samples'] = result.x[1]
  _run.info['optimisation_outcomes'] = list(result['func_vals'])
  _run.info['optimisation_points'] = result['x_iters']
  # run clustering again with best parameters
  out = get_dbscan(eps=result.x[0], min_samples=result.x[1])
  clustering_based_inference(features=out['features'],
                             clustering=out['clustering'],
                             postfix='dbscan')


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
        min_samples=p[1])
    for k in [x for x in out.keys() if not x in ['features', 'clustering']]:
      del out[k]
    cm = torchmetrics.ConfusionMatrix(num_classes=200)
    for frame, pred in clustering_based_inference(features=out['features'],
                                                  postfix='None',
                                                  save=False,
                                                  clustering=out['clustering']):
      label = np.load(os.path.join(directory, f'{frame}_label.npy'))
      label[label >= 37] = 255
      # update confusion matrix, only on labelled pixels
      if np.any(label != 255):
        label = torch.from_numpy(label)
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
    mious.append(measure_assigned_miou(cm))
  _run.info['optimisation_outcomes'] = outcomes
  _run.info['mious'] = mious


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
                             clustering=out['clustering'],
                             postfix='dbscan')


if __name__ == '__main__':
  ex.run_commandline()
