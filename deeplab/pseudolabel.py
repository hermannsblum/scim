import tensorflow_datasets as tfds
from sacred import Experiment
import os
import numpy as np
from tqdm import tqdm
import torchmetrics
import torch
import cv2

import semsegcluster.data.scannet
from semsegcluster.data.nyu_depth_v2 import TRAINING_LABEL_NAMES
from semsegcluster.settings import TMPDIR, EXP_OUT

ex = Experiment()

ex.add_config(
    uncertainty_threshold=-3,
    uncert='pseudolabel-maxlogit-pp',
    inlier='pseudolabel-pred',
)


@ex.main
def create_pseudolabel(subset, pretrained_model, outlier, inlier, uncert,
                       uncertainty_threshold):
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset,
                           str(pretrained_model))
  data = tfds.load(f'scan_net/{subset}', split='validation')
  # compute matching with existing classes
  cm = torchmetrics.ConfusionMatrix(num_classes=200)
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
    # if pose is invalid, prediction and uncertainty cannot be rendered from the map
    try:
      u = np.load(os.path.join(directory, f'{frame}_{uncert}.npy')).squeeze()
    except FileNotFoundError:
      u = np.load(os.path.join(directory, f'{frame}_maxlogit-pp.npy')).squeeze()
    try:
      i = np.load(os.path.join(directory, f'{frame}_{inlier}.npy')).squeeze()
    except FileNotFoundError:
      i = np.load(os.path.join(directory, f'{frame}_pred.npy')).squeeze()
    i[i == np.nan] = 255
    # update confusion matrix, only on labelled pixels
    if np.any(i != 255):
      i = torch.from_numpy(i)
      cluster = np.load(os.path.join(directory,
                                     f'{frame}_{outlier}.npy')).squeeze()
      # make space for wildcard 39
      cluster[cluster >= 39] += 1
      # handle nans as misclassification
      cluster[cluster == -1] = 39
      cluster[cluster == np.nan] = 39
      # cluster numbers larger than 200 are ignored in  the confusion  matrix
      cluster[cluster > 200] = 39
      cluster = torch.from_numpy(cluster)[i != 255]
      u = u[i != 255]
      i = i[i != 255]
      cm.update(cluster, i)
      # cm.update(cluster[u < uncertainty_threshold], i[u < uncertainty_threshold])
  cm = cm.compute().numpy().astype(np.uint32)
  iou = cm[:40].copy()
  iou = iou / (cm[:40].sum(0)[np.newaxis] + cm[:40].sum(1)[:, np.newaxis] -
               cm[:40])
  iou[iou == np.nan] = 0
  matches = np.argwhere(iou > 0.5)
  for match in matches:
    print(
        f'cluster {match[1]} fits to class {TRAINING_LABEL_NAMES[match[0]]} ' +
        f'with {iou[match[0], match[1]]:.2f} IoU')
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
    # if pose is invalid, prediction and uncertainty cannot be rendered from the map
    u_frame = np.load(os.path.join(directory,
                                   f'{frame}_maxlogit-pp.npy')).squeeze()
    try:
      u = np.load(os.path.join(directory, f'{frame}_{uncert}.npy')).squeeze()
    except FileNotFoundError:
      u = u_frame
    u[np.isnan(u)] = u_frame[np.isnan(u)]
    try:
      i = np.load(os.path.join(directory, f'{frame}_{inlier}.npy')).squeeze()
    except FileNotFoundError:
      i = np.load(os.path.join(directory, f'{frame}_pred.npy')).squeeze()
    # outlier are now 0
    i += 1
    i[i == 256] = 0
    o = np.load(os.path.join(directory,
                             f'{frame}_{outlier}.npy')).squeeze() + 41
    o[o == 40] = 0  # ignore outliers
    # apply matching
    for match in matches:
      o[o - 41 == match[1]] = match[0] + 1
    train_signal = i
    train_signal[u > uncertainty_threshold] = o[u > uncertainty_threshold]
    train_signal = train_signal.astype(np.uint16)
    if train_signal.max() < 255:
      train_signal = train_signal.astype(np.uint8)
    cv2.imwrite(
        os.path.join(directory, f'{frame}_merged-{inlier}-{outlier}.png'),
        train_signal)


if __name__ == '__main__':
  ex.run_commandline()
