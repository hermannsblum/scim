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
import sklearn.decomposition
import sklearn.manifold
import sklearn.preprocessing
import sklearn.metrics
import hdbscan
import metaseg_metrics
import matplotlib.pyplot as plt

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

from deeplab.sampling import get_resnet

ex = Experiment()
ex.observers.append(get_observer())

persistent = Memory(EXP_OUT)
memory = Memory("/tmp")


@ex.capture
@persistent.cache
def get_connected_components(subset, pretrained_model, pred_name, uncert_name,
                             uncertainty_threshold):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  anomaly_counter = 1
  anomaly_frames = {}
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
    try:
      pred = np.load(os.path.join(directory,
                                  f'{frame}_{pred_name}.npy')).squeeze()
    except FileNotFoundError:
      pred = np.load(os.path.join(directory, f'{frame}_pred.npy')).squeeze()
    if np.sum(pred != 255) == 0:
      continue
    pred[pred == 255] = 39
    try:
      uncert = np.load(os.path.join(directory,
                                    f'{frame}_{uncert_name}.npy')).squeeze()
    except FileNotFoundError:
      uncert = np.load(os.path.join(directory,
                                    f'{frame}_maxlogit-pp.npy')).squeeze()
    pred_shape = pred.shape
    one_hot = np.eye(40)[pred.reshape((-1))].reshape(
        (pred_shape[0], pred_shape[1], 40))
    _, components = metaseg_metrics.compute_metrics(one_hot, np.ones_like(pred))
    anomaly_pred = np.zeros_like(pred)
    for c in np.unique(components):
      if c < 0:
        continue
      anomaly_pred[components == c] = int(
          uncert[components == c].mean() >= uncertainty_threshold)
      if c > 0:
        # set boundary to same as content
        anomaly_pred[components == -c] = int(
            uncert[components == c].mean() >= uncertainty_threshold)
    if np.sum(anomaly_pred) == 0:
      continue
    _, anomaly_components = metaseg_metrics.compute_metrics(
        np.eye(2)[anomaly_pred.flatten()].reshape(
            (pred_shape[0], pred_shape[1], 2)), np.ones_like(pred))
    anomaly_components[anomaly_pred == 0] = 0
    anomaly_components[anomaly_components < 0] = 0
    for c in np.unique(anomaly_components):
      if c == 0:
        continue
      anomaly_components[anomaly_components == c] = anomaly_counter
      anomaly_counter += 1
    anomaly_frames[frame] = anomaly_components.astype(np.int32)
  return anomaly_frames


@ex.capture
def get_embeddings_of_components(anomaly_frames, subset, feature_name, device):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  model, hooks = get_resnet(feature_name=feature_name, device=device)

  feature_by_component = {}
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
    if frame not in anomaly_frames:
      continue
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)
    # run inference
    _ = model(image)
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])[0]
    assert features.shape[-1] == 2048
    anomaly_components = anomaly_frames[frame]
    # resize to feature size
    anomaly_components = cv2.resize(anomaly_components,
                                    dsize=(features.shape[1],
                                           features.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
    for c in np.unique(anomaly_components):
      if c == 0:
        continue
      feature_by_component[c] = features[anomaly_components == c].mean(0)
      assert feature_by_component[c].shape[0] == 2048
  return feature_by_component


ex.add_config(
    device='cuda',
    feature_name='layer4',
    uncertainty_threshold=-3,
)


@ex.main
def uhlemeyer(_run, pretrained_model, subset, pred_name, eps, min_samples):
  anomaly_frames = get_connected_components()
  feature_by_component = get_embeddings_of_components(
      anomaly_frames=anomaly_frames)
  all_features = np.stack(list(feature_by_component.values()), axis=0)
  print('Loaded all features', flush=True)

  # run PCA
  pca = sklearn.decomposition.PCA(50, copy=False, random_state=1)
  all_features = pca.fit_transform(all_features)
  print('PCA fit', flush=True)

  tsne = sklearn.manifold.TSNE(n_components=2, random_state=2)
  all_features = tsne.fit_transform(all_features)
  print('TSNE fit', flush=True)

  clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                     min_samples=min_samples,
                                     metric='euclidean')
  clustering = clusterer.fit_predict(all_features)
  print(f'Fit clustering, {np.unique(clustering).shape[0]} clusters',
        flush=True)
  plt.scatter(all_features[:, 0], all_features[:, 1], c=clustering)
  plt.savefig('/tmp/tsne_scatter.pdf')
  _run.add_artifact('/tmp/tsne_scatter.pdf')
  plt.show()
  del all_features

  cluster_by_component = {
      k: clustering[i] for i, k in enumerate(feature_by_component.keys())
  }
  del feature_by_component

  # Now run inference
  data = tfds.load(f'scan_net/{subset}', split='validation')
  # make sure the directory exists
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
    try:
      pred = np.load(os.path.join(
          directory, f'{frame}_{pred_name}.npy')).squeeze().astype(np.int32)
    except FileNotFoundError:
      pred = np.load(os.path.join(
          directory, f'{frame}_pred.npy')).squeeze().astype(np.int32)
    if frame in anomaly_frames:
      components = anomaly_frames[frame]
      for c in np.unique(components):
        if c == 0 or c not in cluster_by_component:
          continue
        if cluster_by_component[c] == -1:
          pred[components == c] = -1
        else:
          pred[components == c] = 40 + cluster_by_component[c]
    # save output
    np.save(os.path.join(directory, f'{frame}_uhlemeyer{_run._id}.npy'), pred)


if __name__ == '__main__':
  ex.run_commandline()
