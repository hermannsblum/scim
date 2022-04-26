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

import semseg_density.data.scannet
from semseg_density.data.tfds_to_torch import TFDataIterableDataset
from semseg_density.data.augmentation import augmentation
from semseg_density.data.images import convert_img_to_float
from semseg_density.gdrive import load_gdrive_file
from semseg_density.segmentation_metrics import SegmentationMetric
from semseg_density.settings import TMPDIR, EXP_OUT
from semseg_density.sacred_utils import get_observer, get_checkpoint
from semseg_density.eval import measure_from_confusion_matrix

from deeplab.scannet_dbscan import get_distances, get_model

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")


@ex.capture
@memory.cache
def get_nakajima_distances(subset, shard, subsample, device, pretrained_model,
                           feature_name, pred_name, uncert_name,
                           uncertainty_threshold):
  out = get_distances(subset=subset,
                      shard=shard,
                      subsample=subsample,
                      device=device,
                      pretrained_model=pretrained_model,
                      uncertainty_threshold=uncertainty_threshold,
                      feature_name=feature_name,
                      pred_name=pred_name,
                      uncert_name=uncert_name,
                      normalize=False)
  # weight features by entropy
  out['features'] = np.expand_dims(1 - out.pop('entropy') / np.log(40),
                                   1) * out['features']
  out['distances'] = sp.spatial.distance.pdist(out['features'],
                                               metric='euclidean')
  return out


@ex.capture
def nakajima_inference(weighted_features, clustering, subset, pretrained_model,
                       device, feature_name, _run):
  # set up NN index to find closest clustered sample
  knn = hnswlib.Index(space='l2', dim=256)
  knn.init_index(max_elements=weighted_features.shape[0],
                 M=64,
                 ef_construction=200,
                 random_seed=100)
  knn.add_items(weighted_features, clustering)
  data = tfds.load(f'scan_net/{subset}', split='validation')
  model, hooks = get_model(pretrained_model=pretrained_model,
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
    cluster_pred, _ = knn.knn_query(features, k=1)
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
  del out['inlier_pair']
  del out['same_class_pair']
  del out['same_voxel_pair']
  del out['voxels']
  # add their activation function
  out['distances'] = np.exp(-1.0 * eta * out['distances'])
  # put into square form
  adjacency = sp.spatial.distance.squareform(out.pop('distances'))
  # now run the clustering
  result = mc.run_mcl(adjacency / adjacency.mean(),
                      inflation=inflation,
                      verbose=True)
  clusters = mc.get_clusters(result)
  print(f'Fit clustering to {len(clusters)} clusters', flush=True)
  clustering = -1 * np.ones(out['features'].shape[0], dtype=int)
  for i, cluster in enumerate(clusters):
    for node in cluster:
      clustering[node] = i
  return {
      'features': out['features'],
      'clustering': clustering,
      'prediction': out['prediction'],
      'uncertainty': out['uncertainty'],
  }


@use_named_args(dimensions=mcl_nakajima_dimensions)
def score_mcl_nakajima(eta, inflation):
  out = get_mcl_nakajima(eta=eta, inflation=inflation)
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


ex.add_config(subsample=100,
              shard=5,
              device='cuda',
              uncertainty_threshold=-3,
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
                     clustering=out['clustering'])


@ex.command
def mcl_nakajima():
  out = get_mcl_nakajima()
  nakajima_inference(weighted_features=out['features'],
                     clustering=out['clustering'])


if __name__ == '__main__':
  ex.run_commandline()
