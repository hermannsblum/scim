from sacred import Experiment
import torch
import torchmetrics
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

from semseg_density.data.images import convert_img_to_float
from semseg_density.data.nyu_depth_v2 import TRAINING_LABEL_NAMES
import semseg_density.data.scannet
from semseg_density.model.refinenet import rf_lw50, rf_lw101
from semseg_density.gdrive import load_gdrive_file
from semseg_density.model.refinenet_uncertainty import RefineNetDensity
from semseg_density.settings import TMPDIR, EXP_OUT
from semseg_density.sacred_utils import get_incense_loader

ex = Experiment()


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

@ex.command
def run_refinenet(pretrained_model,
                          subset,
                          n_components,
                          device='cuda',
                          groupnorm=False):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  model = rf_lw50(40, pretrained=False, groupnorm=groupnorm)
  # Load pretrained weights
  if pretrained_model and isinstance(pretrained_model, str):
    checkpoint = torch.load(load_gdrive_file(pretrained_model, ending='pth'))
    load_checkpoint(model, checkpoint, strict=True)
  elif pretrained_model and isinstance(pretrained_model, int):
    loader = get_incense_loader()
    train_exp = loader.find_by_id(pretrained_model)
    train_exp.artifacts['refinenet_scannet_best.pth'].save(TMPDIR)
    checkpoint = torch.load(os.path.join(TMPDIR, f'{pretrained_model}_refinenet_scannet_best.pth'))
    load_checkpoint(model, checkpoint, strict=True)
    pretrained_model = str(pretrained_model)

  model.to(device)
  model.eval()

  # make sure the directory exists, but is empty
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset,
                           pretrained_model)
  os.makedirs(directory, exist_ok=True)
  shutil.rmtree(directory)
  os.makedirs(directory, exist_ok=True)

  cm = torchmetrics.ConfusionMatrix(num_classes=40, compute_on_step=False)

  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy())

    label = tf.cast(blob['labels'], tf.int64)
    # the output is 4 times smaller than the input, so transform labels
    label = tf.image.resize(label[..., tf.newaxis], (120, 160),
                            method='nearest')[..., 0].numpy()

    # run inference
    logits = model(image.to(device))
    pred = torch.argmax(logits, 1)
    max_logit = torch.max(logits, 1)
    softmax_entropy = torch.distributions.categorical.Categorical(
        logits=logits.permute(0, 2, 3, 1)).entropy()

    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      torch_label = torch.from_numpy(label)
      valid_pred = pred[0].detach().to('cpu')[torch_label != 255]
      valid_label = torch_label[torch_label != 255]
      cm.update(valid_pred, valid_label)

    # store outputs
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_entropy.npy'),
            softmax_entropy[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_maxlogit.npy'),
            max_logit[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_label.npy'), label)

  cm = cm.compute().numpy()
  np.save(os.path.join(directory, 'confusion_matrix.npy'), cm)
  disp = ConfusionMatrixDisplay(cm / cm.sum(0),
                                display_labels=TRAINING_LABEL_NAMES)
  plt.figure(figsize=(20, 20))
  disp.plot(ax=plt.gca(), xticks_rotation='vertical', include_values=False)
  plt.savefig(os.path.join(directory, 'confusion_matrix.pdf'))


@ex.main
def run_density_refinenet(pretrained_model,
                          subset,
                          n_components,
                          feature_layer='mflow_conv_g4_pool',
                          size=50,
                          device='cuda',
                          groupnorm=False):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  if not size in [50, 101]:
    raise UserWarning("Unknown model size.")
  model = RefineNetDensity(40,
                           size=size,
                           n_components=n_components,
                           groupnorm=groupnorm,
                           feature_layer=feature_layer)
  # Load pretrained weights
  if pretrained_model and isinstance(pretrained_model, str):
    checkpoint = torch.load(load_gdrive_file(pretrained_model, ending='pth'))
    load_checkpoint(model, checkpoint, strict=False)
  elif pretrained_model and isinstance(pretrained_model, int):
    loader = get_incense_loader()
    train_exp = loader.find_by_id(pretrained_model)
    train_exp.artifacts['refinenet_scannet_density.pth'].save(TMPDIR)
    checkpoint = torch.load(os.path.join(TMPDIR, f'{pretrained_model}_refinenet_scannet_density.pth'))
    load_checkpoint(model, checkpoint, strict=False)
    pretrained_model = str(pretrained_model)

  model.to(device)
  model.eval()

  # make sure the directory exists, but is empty
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset,
                           pretrained_model)
  os.makedirs(directory, exist_ok=True)
  shutil.rmtree(directory)
  os.makedirs(directory, exist_ok=True)

  cm = torchmetrics.ConfusionMatrix(num_classes=40, compute_on_step=False)

  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy())

    label = tf.cast(blob['labels'], tf.int64)
    # the output is 4 times smaller than the input, so transform labels
    label = tf.image.resize(label[..., tf.newaxis], (120, 160),
                            method='nearest')[..., 0].numpy()

    # run inference
    logits, nll = model(image.to(device))
    pred = torch.argmax(logits, 1)
    max_logit = torch.max(logits, 1)
    softmax_entropy = torch.distributions.categorical.Categorical(
        logits=logits.permute(0, 2, 3, 1)).entropy()

    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      torch_label = torch.from_numpy(label)
      valid_pred = pred[0].detach().to('cpu')[torch_label != 255]
      valid_label = torch_label[torch_label != 255]
      cm.update(valid_pred, valid_label)

    # store outputs
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_nll.npy'),
            nll[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_entropy.npy'),
            softmax_entropy[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_maxlogit.npy'),
            max_logit[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_label.npy'), label)

  cm = cm.compute().numpy()
  np.save(os.path.join(directory, 'confusion_matrix.npy'), cm)
  disp = ConfusionMatrixDisplay(cm / cm.sum(0),
                                display_labels=TRAINING_LABEL_NAMES)
  plt.figure(figsize=(20, 20))
  disp.plot(ax=plt.gca(), xticks_rotation='vertical', include_values=False)
  plt.savefig(os.path.join(directory, 'confusion_matrix.pdf'))


if __name__ == '__main__':
  ex.run_commandline()
