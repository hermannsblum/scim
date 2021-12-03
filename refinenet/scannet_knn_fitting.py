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
from semseg_density.model.refinenet import rf_lw50, rf_lw101
from semseg_density.gdrive import load_gdrive_file
from semseg_density.segmentation_metrics import SegmentationMetric
from semseg_density.settings import TMPDIR
from semseg_density.sacred_utils import get_observer, get_incense_loader

ex = Experiment()
ex.observers.append(get_observer())


def load_checkpoint(model, state_dict, strict=True):
  """Load Checkpoint from Google Drive."""
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


@ex.main
def fit(_run,
        subset,
        size=50,
        groupnorm=False,
        metric='cosine',
        batchsize=6,
        subsample=20,
        device='cuda',
        feature_name='mflow_conv_g4_pool',
        pretrained_model='adelaine'):
  # DATA LOADING
  data = tfds.load(f'scan_net/{subset}', split='train', as_supervised=True)
  valdata = data.take(1000)
  traindata = data.skip(1000)

  def data_converter(image, label):
    image = convert_img_to_float(image)
    label = tf.cast(label, tf.int64)
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    # the output is 4 times smaller than the input, so transform labels
    label = tf.image.resize(label[..., tf.newaxis], (120, 160),
                            method='nearest')[..., 0]
    return image, label

  traindata = TFDataIterableDataset(
      traindata.cache().prefetch(10000).map(data_converter))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)

  # MODEL SETUP
  if size == 50:
    model = rf_lw50(40,
                    pretrained=pretrained_model == 'adelaine',
                    groupnorm=groupnorm)
  elif size == 101:
    model = rf_lw101(40,
                     pretrained=pretrained_model == 'adelaine',
                     groupnorm=groupnorm)
  else:
    raise UserWarning("Unknown model size.")
  # Load pretrained weights
  if pretrained_model and pretrained_model != 'adelaine' and isinstance(
      pretrained_model, str):
    checkpoint = torch.load(load_gdrive_file(pretrained_model, ending='pth'))
    load_checkpoint(model, checkpoint, strict=False)
  elif pretrained_model and isinstance(pretrained_model, int):
    loader = get_incense_loader()
    train_exp = loader.find_by_id(pretrained_model)
    train_exp.artifacts['refinenet_scannet_best.pth'].save(TMPDIR)
    checkpoint = torch.load(
        os.path.join(TMPDIR, f'{pretrained_model}_refinenet_scannet_best.pth'))
    load_checkpoint(model, checkpoint, strict=True)
    pretrained_model = str(pretrained_model)
  model.to(device)
  model.eval()

  # Create hook to get features from intermediate pytorch layer
  hooks = {}

  def get_activation(name, features=hooks):

    def hook(model, input, output):
      features['feat'] = output.detach()

    return hook

  # get feature layer
  #feature_name = "mflow_conv_g3_b3_joint_varout_dimred"
  feature_layer = getattr(model, feature_name)
  # register hook to get features
  feature_layer.register_forward_hook(get_activation(feature_name))

  start_time = time.time()

  all_features = []
  for images, labels in tqdm(train_loader):
    images = images.to(device)
    labels = labels.to('cpu').detach()
    out = model(images)
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    # interpolate labels to feature size
    labels = torchvision.transforms.functional.resize(
        labels,
        size=(features.shape[1], features.shape[2]),
        interpolation=PIL.Image.NEAREST)
    for c in np.unique(labels):
      # subsample for each class separately to have a better balance
      if c == 255:
        continue
      class_features = features[labels == c]
      if class_features.shape[0] > subsample:
        # subsampling (because storing all these embeddings would be too much)
        class_features = class_features[np.random.choice(
            class_features.shape[0], size=[subsample], replace=False)]
      all_features.append(class_features)
    del out, labels, images, features, class_features
  all_features = np.concatenate(all_features, axis=0)
  print('Loaded all features', flush=True)

  # set up NN index
  knn = hnswlib.Index(space=metric, dim=256)
  knn.init_index(max_elements=all_features.shape[0],
                 M=64,
                 ef_construction=200,
                 random_seed=100)
  knn.add_items(all_features)
  save_path = os.path.join(TMPDIR, 'knn.pkl')
  with open(save_path, 'wb') as f:
    pickle.dump(knn, f)
  _run.add_artifact(save_path)


if __name__ == '__main__':
  ex.run_commandline()
