from sacred import Experiment
import torch
import torchvision
import torchmetrics
import PIL
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import cv2

from refinenet.scannet_density_fitting import train

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
from semseg_density.model.refinenet_sml import RefineNetSML
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
        pretrained_model,
        size=50,
        groupnorm=False,
        batchsize=6,
        datacache=False,
        device='cuda'):
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

  traindata = traindata.map(data_converter)
  if datacache:
    traindata = traindata.cache().prefetch(10000)
  traindata = TFDataIterableDataset(traindata)
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
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])
  model.eval()

  means = {c: torchmetrics.MeanMetric(compute_on_step=False) for c in range(40)}

  start_time = time.time()

  for images, labels in tqdm(train_loader):
    images = images.to(device)
    labels = labels.detach()
    out = model(images).permute([0, 2, 3, 1]).detach()
    for c in torch.unique(labels).numpy():
      if c == 255:
        continue
      class_logits = out[labels == c].to('cpu')
      means[c].update(class_logits)
      del class_logits
    del out, labels, images
  computed_means = torch.cat([means[c].compute() for c in range(40)]).to(device)
  del means
  print('Computed Means', computed_means, flush=True)

  vars = {c: torchmetrics.MeanMetric(compute_on_step=False) for c in range(40)}
  for images, labels in tqdm(train_loader):
    images = images.to(device)
    labels = labels.detach()
    out = model(images).permute([0, 2, 3, 1]).detach()
    mse = torch.square(out - computed_means)
    for c in np.unique(labels):
      if c == 255:
        continue
      class_mse = mse[labels == c].to('cpu')
      vars[c].update(class_mse)
      del class_mse
    del out, labels, images, mse
  computed_vars = torch.cat([vars[c].compute() for c in range(40)]).to('cpu')
  print('Computed Vars', computed_vars, flush=True)

  smlmodel = RefineNetSML(40,
                          size=size,
                          pretrained=False,
                          groupnorm=groupnorm,
                          means=computed_means.to('cpu'),
                          vars=computed_vars)
  smlmodel.to(device)
  smlmodel.eval()
  load_checkpoint(smlmodel.refinenet, checkpoint, strict=True)

  # testing
  images, labels = next(iter(train_loader))
  logit, sml = smlmodel(images.to(device))

  # saving
  filename = 'refinenet_scannet_sml.pth'
  save_path = os.path.join(TMPDIR, filename)
  torch.save(smlmodel.state_dict(), save_path)
  _run.add_artifact(save_path)


if __name__ == '__main__':
  ex.run_commandline()
