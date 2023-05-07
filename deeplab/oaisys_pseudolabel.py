import tensorflow_datasets as tfds
from sacred import Experiment
import os
import numpy as np
from tqdm import tqdm
import torchmetrics
import torch
import cv2

from semsegcluster.settings import EXP_OUT
from oaisys_inference import OAISYS_LABELS

ex = Experiment()

ex.add_config(
    uncertainty_threshold=-3,
    uncert='pseudolabel-maxlogit-pp',
    inlier='pseudolabel-pred',
)


@ex.main
def create_pseudolabel(subset, outlier, inlier, uncert,
                       uncertainty_threshold):
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, 'deeplab')
  data = tfds.load(f'{subset}', split='validation')
  # compute matching with existing classes
  cm = torchmetrics.ConfusionMatrix(num_classes=200)
  for idx in tqdm(range(len(data))):
    frame = f'{idx:06d}'
    # if pose is invalid, prediction and uncertainty cannot be rendered from the map
    try:
      u = np.load(os.path.join(directory, f'{frame}_{uncert}.npy')).squeeze()
    except FileNotFoundError:
      try:
        u = np.load(os.path.join(directory, f'{idx}_{uncert}.npy')).squeeze()
      except FileNotFoundError:
        print('file not found: ', f'{frame}/{idx}_{uncert}.npy', 'using maxlogit-pp instead')
        u = np.load(os.path.join(directory, f'{idx}_maxlogit-pp.npy')).squeeze()
    try:
      i = np.load(os.path.join(directory, f'{frame}_{inlier}.npy')).squeeze()
    except FileNotFoundError:
      try:
        i = np.load(os.path.join(directory, f'{idx}_{inlier}.npy')).squeeze()
      except FileNotFoundError:
        print('file not found: ', f'{frame}/{idx}_{inlier}.npy', 'using pred instead')
        i = np.load(os.path.join(directory, f'{idx}_pred.npy')).squeeze()
    i[i == np.nan] = 255
    # update confusion matrix, only on labelled pixels
    if np.any(i != 255):
      i = torch.from_numpy(i)
      cluster = np.load(os.path.join(directory,
                                     f'{idx:06d}_{outlier}.npy')).squeeze()
      # make space for wildcard 11
      cluster[cluster >= 11] += 1
      # handle nans as misclassification
      cluster[cluster == -1] = 11
      cluster[cluster == np.nan] = 11
      # cluster numbers larger than 200 are ignored in  the confusion  matrix
      cluster[cluster > 200] = 11
      cluster = torch.from_numpy(cluster)[i != 255]
      u = u[i != 255]
      i = i[i != 255]
      cm.update(cluster, i)
  cm = cm.compute().numpy().astype(np.uint32)
  iou = cm[:15].copy()
  iou = iou / (cm[:15].sum(0)[np.newaxis] + cm[:15].sum(1)[:, np.newaxis] -
               cm[:15])
  iou[iou == np.nan] = 0
  matches = np.argwhere(iou > 0.5)
  for match in matches:
    print(
        f'cluster {match[1]} fits to class {OAISYS_LABELS[match[0]]} ' +
        f'with {iou[match[0], match[1]]:.2f} IoU')
  for idx in tqdm(range(len(data))):
    frame = f'{idx:06d}'
    # if pose is invalid, prediction and uncertainty cannot be rendered from the map
    if os.path.exists(os.path.join(directory, f'{idx}_maxlogit-pp.npy')):
      u_frame = np.load(os.path.join(directory,
                                    f'{idx}_maxlogit-pp.npy')).squeeze()
    else:
      u_frame = None
    try:
      u = np.load(os.path.join(directory, f'{frame}_{uncert}.npy')).squeeze()
    except FileNotFoundError:
      try:
        u = np.load(os.path.join(directory, f'{idx}_{uncert}.npy')).squeeze()
      except FileNotFoundError:
        print('file not found: ', f'{frame}/{idx}_{uncert}.npy', 'using maxlogit-pp instead')
        u = u_frame
    if u_frame is not None:
      u[np.isnan(u)] = u_frame[np.isnan(u)]
    try:
      i = np.load(os.path.join(directory, f'{frame}_{inlier}.npy')).squeeze()
    except FileNotFoundError:
      try:
        i = np.load(os.path.join(directory, f'{idx}_{inlier}.npy')).squeeze()
      except FileNotFoundError:
        print('file not found: ', f'{frame}/{idx}_{inlier}.npy', 'using pred instead')
        i = np.load(os.path.join(directory, f'{idx}_pred.npy')).squeeze()
    o = np.load(os.path.join(directory,
                             f'{idx:06d}_{outlier}.npy')).squeeze() + 12
    # apply matching
    for match in matches:
      o[o - 12 == match[1]] = match[0]
    train_signal = i
    train_signal[u > uncertainty_threshold] = o[u > uncertainty_threshold]
    train_signal = train_signal.astype(np.uint16)
    if train_signal.max() < 255:
      train_signal = train_signal.astype(np.uint8)
    cv2.imwrite(
        os.path.join(directory, f'{idx:06d}_merged-{inlier}-{outlier}_{uncertainty_threshold}.png'),
        train_signal)


if __name__ == '__main__':
  ex.run_commandline()
