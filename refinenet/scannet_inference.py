from sacred import Experiment
import torch
import torchmetrics
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import cv2
import shutil
import pickle
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
from semseg_density.model.refinenet_sml import RefineNetSML
from semseg_density.settings import TMPDIR, EXP_OUT
from semseg_density.sacred_utils import get_incense_loader, get_checkpoint

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
def run_knn(fitting_experiment,
            subset,
            device='cuda',
            groupnorm=False,
            ef=100,
            k=1,
            feature_name='mflow_conv_g4_pool'):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  loader = get_incense_loader()
  fitting_exp = loader.find_by_id(fitting_experiment)
  # MODEL SETUP
  model = rf_lw50(40, pretrained=False, groupnorm=groupnorm)
  checkpoint, pretrained_id = get_checkpoint(
      fitting_exp.config.pretrained_model)
  load_checkpoint(model, checkpoint)
  fitting_exp.artifacts['knn.pkl'].save(TMPDIR)
  with open(os.path.join(TMPDIR, f'{fitting_experiment}_knn.pkl'), 'rb') as f:
    knn = pickle.load(f)
  knn.set_ef(ef)
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

  # make sure the directory exists
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)

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
    out = model(image.to(device))
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    feature_shape = features.shape

    # query knn
    _, distances = knn.knn_query(features.reshape([-1, 256]), k=k)
    if k > 1:
      distances = distances.mean(1)
    distances = distances.reshape(feature_shape[1:3])
    if feature_shape[1] != 120 or feature_shape[2] != 160:
      distances = cv2.resize(distances, (160, 120),
                             interpolation=cv2.INTER_LINEAR)

    # store outputs
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_{k}nn_{fitting_experiment}.npy'),
            distances)


@ex.command
def run_refinenet(pretrained_model,
                  subset,
                  device='cuda',
                  groupnorm=False,
                  ignore_other=False):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  # MODEL SETUP
  model = rf_lw50(40, pretrained=False, groupnorm=groupnorm)
  checkpoint, pretrained_id = get_checkpoint(pretrained_model)
  load_checkpoint(model, checkpoint)
  pretrained_model = str(pretrained_model)

  model.to(device)
  model.eval()

  # make sure the directory exists
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
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
    if ignore_other:
      label[label >= 37] = 255

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
            -max_logit[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_label.npy'), label)

    if 'instances' in blob:
      instances = tf.cast(blob['instances'], tf.int64)
      # the output is 4 times smaller than the input, so transform labels
      instances = tf.image.resize(instances[..., tf.newaxis], (120, 160),
                              method='nearest')[..., 0].numpy()
      np.save(os.path.join(directory, f'{name}_instances.npy'), instances)

  cm = cm.compute().numpy()
  np.save(os.path.join(directory, 'confusion_matrix.npy'), cm)
  disp = ConfusionMatrixDisplay(cm / cm.sum(0),
                                display_labels=TRAINING_LABEL_NAMES)
  plt.figure(figsize=(20, 20))
  disp.plot(ax=plt.gca(), xticks_rotation='vertical', include_values=False)
  plt.savefig(os.path.join(directory, 'confusion_matrix.pdf'))


@ex.main
def run_sml_refinenet(pretrained_model,
                      subset,
                      size=50,
                      device='cuda',
                      groupnorm=False):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  if not size in [50, 101]:
    raise UserWarning("Unknown model size.")
  model = RefineNetSML(
      40,
      size=size,
      groupnorm=groupnorm,
  )
  # Load pretrained weights
  checkpoint, pretrained_id = get_checkpoint(
      pretrained_model, pthname='refinenet_scannet_sml.pth')
  load_checkpoint(model, checkpoint)
  model.to(device)
  model.eval()

  # make sure the directory exists, but is empty
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)
  shutil.rmtree(directory)
  os.makedirs(directory, exist_ok=True)

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
    _, sml = model(image.to(device))

    # store outputs
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_sml.npy'),
            sml[0].detach().to('cpu').numpy())


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
  checkpoint, pretrained_id = get_checkpoint(
      pretrained_model, pthname='refinenet_scannet_density.pth')
  load_checkpoint(model, checkpoint)

  model.to(device)
  model.eval()

  # make sure the directory exists, but is empty
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
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
