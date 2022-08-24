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


@ex.main
def run_deeplab(training,
                subset,
                split='validation',
                device='cuda',
                ignore_other=True):
  data = tfds.load(f'scan_net/{subset}', split=split)

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=50,
      aux_loss=None)
  checkpoint, _ = get_checkpoint(training)
  # remove any aux classifier stuff
  removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
  for k in removekeys:
    del checkpoint[k]
  load_checkpoint(model, checkpoint)

  loader = get_incense_loader()
  fitting_exp = loader.find_by_id(training)
  pretrained_id = str(fitting_exp.config.pretrained_model)
  pseudolabels = fitting_exp.config.pseudolabels

  model.to(device)
  model.eval()
  postprocessing = BoundarySuppressionWithSmoothing().to(device)

  # make sure the directory exists
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  os.makedirs(directory, exist_ok=True)
  for blob in tqdm(data):
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)

    # run inference
    logits = model(image)['out']
    max_logit, pred = torch.max(logits, 1)
    max_logit = postprocessing(max_logit, prediction=pred)

    # store outputs
    name = blob['name'].numpy().decode()
    out = pred[0].detach().to('cpu').numpy().astype(np.uint8)
    # 0 is magic number for outlier or nan
    out += 1
    cv2.imwrite(
        os.path.join(directory, f'{name}_pred{training}-{pseudolabels}.png'),
        out)
    #np.save(os.path.join(directory, f'{name}_pred{training}-{pseudolabels}.npy'),
    #       pred[0].detach().to('cpu').numpy().astype('uint8'))
    #np.save(os.path.join(directory, f'{name}_maxlogit-pp{training}-{pseudolabels}.npy'),
    #        -max_logit[0].detach().to('cpu').numpy())


if __name__ == '__main__':
  ex.run_commandline()
