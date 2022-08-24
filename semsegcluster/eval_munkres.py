import torchmetrics
import numpy as np
import torch
import sklearn
import sklearn.metrics
import os
import tensorflow as tf
import cv2

tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
from tqdm import tqdm
from joblib import Memory
from munkres import Munkres, DISALLOWED

from semsegcluster.settings import EXP_OUT
from semsegcluster.data.nyu_depth_v2 import TRAINING_LABEL_NAMES

memory = Memory(EXP_OUT)


def measure_from_confusion_matrix(cm, is_prediction=False, beta=1.0):
  assert np.sum(cm[40:]) == 0
  cm = cm[:40]

  m = Munkres()
  cost = (cm.max() + 1) - cm
  cost = cost.tolist()
  if is_prediction:
    # class predictions must not be matched to other classes
    for i in range(40):
      for j in range(i):
        cost[i][j + 1] = DISALLOWED
        cost[j][i + 1] = DISALLOWED
  # prediction 0 is our wildcard for wrong classifications, must not be matched
  for i in range(40):
    cost[i][0] = DISALLOWED
  assigned_idx = m.compute(cost)
  iou = np.zeros(40, dtype=np.float32)
  for label, cluster in assigned_idx:
    if label >= 40:
      continue
    iou[label] = cm[label, cluster] / (cm[label].sum() + cm[:, cluster].sum() -
                                       cm[label, cluster])
  measurements = {
      'assigned_iou': iou,
      'assigned_miou': np.nanmean(iou),
      'assignment': assigned_idx,
      'confusion_matrix': cm,
  }
  # contingency matrix based sklearn metrics
  # taken from https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/metrics/cluster/_supervised.py#L402
  cmf = cm.astype(np.float64)
  n_total = cmf.sum()
  n_labels = cmf.sum(1)
  n_labels = n_labels[n_labels > 0]
  entropy_labels = -np.sum(
      (n_labels / n_total) * (np.log(n_labels) - np.log(n_total)))
  n_pred = cmf.sum(0)
  n_pred = n_pred[n_pred > 0]
  entropy_pred = -np.sum(
      (n_pred / n_total) * (np.log(n_pred) - np.log(n_total)))
  mutual_info = sklearn.metrics.mutual_info_score(None, None, contingency=cm)
  homogeneity = mutual_info / (entropy_labels) if entropy_labels else 1.0
  completeness = mutual_info / (entropy_pred) if entropy_pred else 1.0
  if homogeneity + completeness == 0.0:
    v_measure_score = 0.0
  else:
    v_measure_score = ((1 + beta) * homogeneity * completeness /
                       (beta * homogeneity + completeness))
  measurements.update({
      'homogeneity': homogeneity,
      'completeness': completeness,
      'v_score': v_measure_score,
  })
  return measurements


@memory.cache
def get_npy_measurements_of_method_with_munkres(subset,
                                                pretrained_id,
                                                method,
                                                ignore_other=True):
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  cm = torchmetrics.ConfusionMatrix(num_classes=200)
  if subset == 'scene0598_02':
    # ignore class bookshelf
    bookshelf = next(
        i for i in range(40) if TRAINING_LABEL_NAMES[i] == 'bookshelf')
  data = tfds.load(f'scan_net/{subset}', split='validation')
  is_prediction = ('pred' in method or 'uhlemeyer' in method)
  for blob in tqdm(data):
    label = tf.cast(blob['labels_nyu'], tf.int64).numpy()
    frame = blob['name'].numpy().decode()
    if ignore_other:
      label[label >= 37] = 255
    if subset == 'scene0598_02':
      label[label == bookshelf] = 255
    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      label = torch.from_numpy(label)
      pred = np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
      # make space for wildcard cluster 0
      pred += 1
      # handle nans as misclassification
      if is_prediction:
        pred[pred == 40] = 0  # old code 39
      pred[np.isnan(pred)] = 0
      # cluster numbers larger than 200 are ignored in the confusion matrix
      pred[pred > 200] = 0
      pred = torch.from_numpy(pred)[label != 255]
      label = label[label != 255]
      cm.update(pred, label)
  cm = cm.compute().numpy().astype(np.uint32)
  return measure_from_confusion_matrix(cm, is_prediction=is_prediction)


@memory.cache
def get_png_measurements_of_method_with_munkres(subset,
                                                pretrained_id,
                                                method,
                                                ignore_other=True):
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  cm = torchmetrics.ConfusionMatrix(num_classes=200)
  if subset == 'scene0598_02':
    # ignore class bookshelf
    bookshelf = next(
        i for i in range(40) if TRAINING_LABEL_NAMES[i] == 'bookshelf')
  data = tfds.load(f'scan_net/{subset}', split='validation')
  is_prediction = ('pred' in method or 'uhlemeyer' in method)
  for blob in tqdm(data):
    label = tf.cast(blob['labels_nyu'], tf.int64).numpy()
    frame = blob['name'].numpy().decode()
    if ignore_other:
      label[label >= 37] = 255
    if subset == 'scene0598_02':
      label[label == bookshelf] = 255
    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      label = torch.from_numpy(label)
      pred = cv2.imread(os.path.join(directory, f'{frame}_{method}.png'),
                        cv2.IMREAD_ANYDEPTH).squeeze()
      # handle nans as misclassification
      pred[np.isnan(pred)] = 0
      # cluster numbers larger than 200 are ignored in  the confusionm  matrix
      pred[pred > 200] = 0
      pred = torch.from_numpy(pred)[label != 255]
      label = label[label != 255]
      cm.update(pred, label)
  cm = cm.compute().numpy().astype(np.uint32)
  return measure_from_confusion_matrix(cm, is_prediction=is_prediction)


def get_measurements(subset, pretrained_id, ignore_other=True):
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  first_frame = f'{subset}_000000'
  methods = list(
      set(
          filename.split(first_frame)[-1].split('.')[0][1:]
          for filename in os.listdir(directory)
          if filename.startswith(first_frame)))
  measurements = {}
  for method in methods:
    if os.path.exists(os.path.join(directory, f'{first_frame}_{method}.png')):
      measurements[method] = get_png_measurements_of_method_with_munkres(
          subset, pretrained_id, method, ignore_other=ignore_other)
      continue
    pred = np.load(os.path.join(directory, f'{first_frame}_{method}.npy')).squeeze()
    if not np.issubdtype(pred.dtype, np.integer) or pred.dtype == np.uint32:
      #print(f'Ignoring {method} because data is not integer type.')
      continue
    if not pred.shape[0] == 480:
      #print(f'Ignoring {method} because shape {pred.shape} does not match.')
      continue
    measurements[method] = get_npy_measurements_of_method_with_munkres(
        subset, pretrained_id, method, ignore_other=ignore_other)
  return measurements
