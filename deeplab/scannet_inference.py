from sacred import Experiment
import torch
import torchvision
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

import semsegcluster.data.scannet
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.data.nyu_depth_v2 import TRAINING_LABEL_NAMES
from semsegcluster.model.deeplab_sml import DeeplabSML
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_incense_loader, get_checkpoint
from semsegcluster.model.postprocessing import BoundarySuppressionWithSmoothing

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
def run_sml(fitting_experiment, subset, device='cuda'):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  model = DeeplabSML(40)
  checkpoint, pretrained_id = get_checkpoint(fitting_experiment)
  load_checkpoint(model, checkpoint)
  model.to(device)
  model.eval()

  # make sure the directory exists
  loader = get_incense_loader()
  fitting_exp = loader.find_by_id(fitting_experiment)
  pretrained_id = str(fitting_exp.config.pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)

  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)

    # run inference
    logits, sml = model(image)

    # store outputs
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_sml.npy'),
            sml.detach().to('cpu').numpy())


@ex.command
def run_knn(
    fitting_experiment,
    subset,
    device='cuda',
    ef=200,
    k=1,
    feature_name='classifier.2',
):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  loader = get_incense_loader()
  fitting_exp = loader.find_by_id(fitting_experiment)
  fitting_exp.artifacts['knn.pkl'].save(TMPDIR)
  with open(os.path.join(TMPDIR, f'{fitting_experiment}_knn.pkl'), 'rb') as f:
    knn = pickle.load(f)
  knn.set_ef(ef)
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=40,
      aux_loss=None)
  checkpoint, pretrained_id = get_checkpoint(
      fitting_exp.config.pretrained_model)
  # remove any aux classifier stuff
  removekeys = [
      key for key in checkpoint.keys() if key.startswith('aux_classifier')
  ]
  for key in removekeys:
    del checkpoint[key]
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

  # make sure the directory exists
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)

  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)

    # run inference
    out = model(image)
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    feature_shape = features.shape

    # query knn
    _, distances = knn.knn_query(features.reshape([-1, 256]), k=k)
    if k > 1:
      distances = distances.mean(1)
    distances = distances.reshape(feature_shape[1:3])
    if feature_shape[1] != 640 or feature_shape[2] != 480:
      distances = cv2.resize(distances, (640, 480),
                             interpolation=cv2.INTER_LINEAR)

    # store outputs
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_{k}nn_{fitting_experiment}.npy'),
            distances)

    del out, image, features, distances


@ex.main
def run_deeplab(pretrained_model, subset, device='cuda', ignore_other=True):
  if subset == 'val100' or subset.startswith('scene'):
    data = tfds.load(f'scan_net/{subset}', split='validation')
  else:
    data = tfds.load(f'scan_net/{subset}', split='test')

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=40,
      aux_loss=None)
  checkpoint, pretrained_id = get_checkpoint(pretrained_model)
  # remove any aux classifier stuff
  removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
  for k in removekeys:
    del checkpoint[k]
  load_checkpoint(model, checkpoint)
  pretrained_model = str(pretrained_model)

  model.to(device)
  model.eval()
  postprocessing = BoundarySuppressionWithSmoothing().to(device)

  # make sure the directory exists
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)

  cm = torchmetrics.ConfusionMatrix(num_classes=40, compute_on_step=False)

  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)

    label = tf.cast(blob['labels_nyu'], tf.int64).numpy()
    if ignore_other:
      label[label >= 37] = 255

    # run inference
    logits = model(image)['out']
    max_logit, pred = torch.max(logits, 1)
    softmax_entropy = torch.distributions.categorical.Categorical(
        logits=logits.permute(0, 2, 3, 1)).entropy()
    max_logit = postprocessing(max_logit, prediction=pred)

    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      torch_label = torch.from_numpy(label)
      valid_pred = pred[0].detach().to('cpu')[torch_label != 255]
      valid_label = torch_label[torch_label != 255]
      cm.update(valid_pred, valid_label)

    # store outputs
    name = blob['name'].numpy().decode()
    np.save(os.path.join(directory, f'{name}_pred.npy'),
            pred[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_entropy.npy'),
            softmax_entropy[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_maxlogit-pp.npy'),
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


if __name__ == '__main__':
  ex.run_commandline()
