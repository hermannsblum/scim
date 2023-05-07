from collections import OrderedDict
from sacred import Experiment
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)
import cv2
import numpy as np
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')

from semsegcluster.data.images import convert_img_to_float
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_checkpoint

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


@ex.main
def run_deeplab(training,
                pretrained_models,
                subset,
                pseudolabels,
                device='cuda',
                ignore_other=True,
                use_euler=False):
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{subset}.tar')
    data = tfds.load(f'{subset}', split='validation', data_dir=f'{TMPDIR}/datasets')
    pretrained_models = []
    for i in range(10):
      pretrained_models.append(f'/cluster/scratch/loewsi/scimfolder/logs/{training}/deeplab_oaisys_adapted_best_model_{i}.pth')
  else:
    data = tfds.load(f'{subset}', split='validation')

  # MODEL SETUP
  models = []
  for pretrained_model in pretrained_models:
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        pretrained_backbone=False,
        progress=True,
        num_classes=21,
        aux_loss=None)
    checkpoint, pretrained_id = get_checkpoint(pretrained_model)
    # remove any aux classifier stuff
    removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
    for k in removekeys:
      del checkpoint[k]
    load_checkpoint(model, checkpoint)

    model.to(device)
    model.eval()
    models.append(model)

  # make sure the directory exists
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)
  for idx, blob in tqdm(enumerate(data)):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)

    # run inference
    for i, model in enumerate(models):
      logits = model(image.to(device))['out']
      if i == 0:
        logits_sum = logits
      else:
        logits_sum += logits
    logits = logits_sum / len(models)
    _, pred = torch.max(logits, 1)

    # store outputs
    out = pred[0].detach().to('cpu').numpy().astype(np.uint8)
    cv2.imwrite(
        os.path.join(directory, f'{idx:06d}_pred{training}-{pseudolabels}.png'),
        out)


if __name__ == '__main__':
  ex.run_commandline()
