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
import hdbscan

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
def get_embeddings(subset, shard, subsample, device, pretrained_model, feature_name):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  model, hooks = get_model(pretrained_model, feature_name, device)

  all_features = []
  for blob in tqdm(data.shard(shard, 0)):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)
    # run inference
    logits = model(image)['out']
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    features = features.reshape((-1, 256))
    # subsampling (because storing all these embeddings would be too much)
    features = features[np.random.choice(
        features.shape[0], size=[subsample], replace=False)]
    all_features.append(features)

    del logits,  image, features
  return np.concatenate(all_features, axis=0)


def clustering_evaluation(cluster, labels, instances):
  ret = {}
  ret['homogeneity'] = sklearn.metrics.homogeneity_score(labels, cluster)
  # match clusters to classes
  clusteredlabel = np.zeros_like(labels)
  for c in np.unique(cluster):
    # assign to most observed label
    assigned, counts = np.unique(labels[cluster == c], return_counts=True)
    clusteredlabel[cluster == c] = assigned[np.argmax(counts)]
  ret['accuracy'] = sklearn.metrics.accuracy_score(labels, clusteredlabel)
  # assign each instance to the majority cluster
  inst_ids = []
  inst_clusters = []
  inst_labels = []
  for i in sorted(np.unique(instances)):
    inst_ids.append(i)
    assigned, counts = np.unique(cluster[instances == i], return_counts=True)
    inst_clusters.append(assigned[np.argmax(counts)])
    assigned, counts = np.unique(labels[instances == i], return_counts=True)
    inst_labels.append(assigned[np.argmax(counts)])
  inst_ids = np.array(inst_ids)
  inst_clusters = np.array(inst_clusters)
  inst_labels = np.array(inst_labels)
  ret['inst_homogeneity'] = sklearn.metrics.homogeneity_score(
      inst_labels, inst_clusters)
  # match clusters to classes
  inst_clusteredlabel = np.zeros_like(inst_labels)
  for c in np.unique(inst_clusters):
    # assign to most observed label
    assigned, counts = np.unique(inst_labels[inst_clusters == c],
                                 return_counts=True)
    inst_clusteredlabel[inst_clusters == c] = assigned[np.argmax(counts)]
  ret['inst_accuracy'] = sklearn.metrics.accuracy_score(inst_labels,
                                                        inst_clusteredlabel)
  return ret


ex.add_config(
    subsample=100,
    shard=5,
    algorithm='full',
    n_clusters=60,
    device='cuda',
    feature_name='classifier.2',
    ignore_other=True,
)

@ex.main
def kmeans(_run, algorithm, n_clusters, ignore_other, pretrained_model, device,
           normalize, subset):
  features= get_embeddings()
  print('Loaded all features', flush=True)

  clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                     copy_x=False,
                                     algorithm=algorithm)
  if normalize:
    features = sklearn.preprocessing.normalize(features, norm='l2')
  clustering = clusterer.fit_predict(features)
  _run.info['clustering'] = clustering
  print('Fit clustering', flush=True)

  # Now run inference
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
    cluster_pred = clusterer.predict(features.reshape((-1, 256)))
    cluster_pred = cluster_pred.reshape((feature_shape[1], feature_shape[2]))
    cluster_pred = cv2.resize(cluster_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_kmeans{_run._id}.npy'),
            cluster_pred)


if __name__ == '__main__':
  ex.run_commandline()
