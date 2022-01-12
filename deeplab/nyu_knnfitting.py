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
from semseg_density.gdrive import load_gdrive_file
from semseg_density.segmentation_metrics import SegmentationMetric
from semseg_density.settings import TMPDIR
from semseg_density.sacred_utils import get_observer, get_checkpoint

ex = Experiment()
ex.observers.append(get_observer())


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


ex.add_config(
    metric='cosine',
    batchsize=6,
    subsample=20,
    device='cuda',
    feature_name='classifier.2',
    ignore_other=True,
)


@ex.main
def fit(
    _run,
    subset,
    metric,
    batchsize,
    subsample,
    device,
    feature_name,
    ignore_other,
    pretrained_model,
):
  # DATA LOADING
  data = tfds.load(f'nyu_depth_v2_labeled/{subset}', split='train', as_supervised=True)
  valdata = data.take(500)
  traindata = data.skip(500)

  def data_converter(image, label):
    image = convert_img_to_float(image)
    label = tf.cast(label, tf.int64)
    if ignore_other:
      label = tf.where(label >= 37, tf.cast(255, tf.int64), label)
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

  traindata = TFDataIterableDataset(
      traindata.cache().prefetch(10000).map(data_converter))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)

  # MODEL SETUP
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

  start_time = time.time()

  all_features = []
  for images, labels in tqdm(train_loader):
    images = images.to(device)
    labels = labels.to('cpu').detach()
    _ = model(images)
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
