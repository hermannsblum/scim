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

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile

import semseg_density.data.nyu_depth_v2
from semseg_density.data.tfds_to_torch import TFDataIterableDataset
from semseg_density.data.augmentation import augmentation
from semseg_density.data.images import convert_img_to_float
from semseg_density.model.deeplab_sml import DeeplabSML
from semseg_density.settings import TMPDIR
from semseg_density.sacred_utils import get_observer, get_checkpoint

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
        batchsize=6,
        datacache=False,
        device='cuda'):
  # DATA LOADING
  data = tfds.load(f'nyu_depth_v2_labeled/{subset}',
                   split='train',
                   as_supervised=True)
  valdata = data.take(500)
  traindata = data.skip(500)

  def data_converter(image, label):
    image = convert_img_to_float(image)
    label = tf.cast(label, tf.int64)
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

  if datacache:
    traindata = traindata.cache().prefetch(10000)
  traindata = TFDataIterableDataset(traindata.map(data_converter))
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
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])
  model.eval()

  means = {c: torchmetrics.MeanMetric(compute_on_step=False) for c in range(40)}

  start_time = time.time()

  for images, labels in tqdm(train_loader):
    images = images.to(device)
    logits = model(images)['out'].permute([0, 2, 3, 1]).detach()
    pred = torch.argmax(logits, 3).detach().cpu()
    for c in torch.unique(pred).numpy():
      class_logit = logits[pred == c].mean().to('cpu')
      means[c].update(class_logit, weight=torch.sum(pred == c))
      del class_logit
    del logits, labels, images
  computed_means = torch.cat([means[c].compute() for c in range(40)]).to(device)
  del means
  print('Computed Means', computed_means, flush=True)

  vars = {c: torchmetrics.MeanMetric(compute_on_step=False) for c in range(40)}
  for images, labels in tqdm(train_loader):
    images = images.to(device)
    logits = model(images)['out'].permute([0, 2, 3, 1]).detach()
    max_logit, pred = torch.max(logits, 3)
    mse = torch.square(max_logit - torch.take(computed_means, pred))
    pred = pred.cpu()
    for c in torch.unique(pred).numpy():
      class_mse = mse[pred == c].mean().to('cpu')
      vars[c].update(class_mse, weight=torch.sum(pred == c))
      del class_mse
    del logits, labels, images, mse
  computed_vars = torch.cat([vars[c].compute() for c in range(40)]).to('cpu')
  print('Computed Vars', computed_vars, flush=True)

  smlmodel = DeeplabSML(40, means=computed_means.to('cpu'), vars=computed_vars)
  smlmodel.to(device)
  smlmodel.eval()
  load_checkpoint(smlmodel.deeplab, checkpoint, strict=True)

  # testing
  images, labels = next(iter(train_loader))
  logit, sml = smlmodel(images.to(device))

  # saving
  filename = 'deeplab_nyu_sml.pth'
  save_path = os.path.join(TMPDIR, filename)
  torch.save(smlmodel.state_dict(), save_path)
  _run.add_artifact(save_path)


if __name__ == '__main__':
  ex.run_commandline()
