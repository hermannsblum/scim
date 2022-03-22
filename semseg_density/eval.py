import torchmetrics
import numpy as np
import torch
import os
from tqdm import tqdm
from joblib import Memory

from semseg_density.settings import EXP_OUT

memory = Memory("/tmp")


def measure_from_confusion_matrix(cm, is_prediction=False):
  assert np.sum(cm[40:]) == 0
  cm = cm[:40]
  newcm = np.zeros((40, 40), dtype=np.uint32)
  for pred_c in range(cm.shape[1]):
    if is_prediction and pred_c < 40:
      assigned_class = pred_c
    elif pred_c == 39:
      assigned_class = pred_c
    else:
      assigned_class = np.argmax(cm[:, pred_c])
    newcm[:, assigned_class] += cm[:, pred_c]
  iou = np.diag(newcm) / (newcm.sum(0) + newcm.sum(1) - np.diag(newcm))
  measurements = {
      'confusion_matrix': cm,
      'accumulated_confusion_matrix': newcm,
      'accumulated_iou': iou,
      'accumulated_miou': np.nanmean(iou),
  }
  newcm = np.zeros((40, 40), dtype=np.uint32)
  for pred_c in range(cm.shape[1]):
    if is_prediction and pred_c < 40:
      assigned_class = pred_c
    elif pred_c == 39:
      assigned_class = 38
    else:
      assigned_class = np.argmax(cm[:, pred_c])
    if newcm[assigned_class, assigned_class] < cm[assigned_class, pred_c]:
      # count those existing assigned classifications as misclassifications
      newcm[:, 38] += newcm[:, assigned_class]
      # assign current cluster to this class
      newcm[:, assigned_class] = cm[:, pred_c]
    else:
      newcm[:, 38] += cm[:, pred_c]
  iou = np.diag(newcm) / (newcm.sum(0) + newcm.sum(1) - np.diag(newcm))
  measurements.update({
      'assigned_confusion_matrix': newcm,
      'assigned_iou': iou,
      'assigned_miou': np.nanmean(iou),
  })
  return measurements


@memory.cache
def get_measurements_of_method(path, method, ignore_other=True):
  directory = os.path.join(EXP_OUT, path)
  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  cm = torchmetrics.ConfusionMatrix(num_classes=200)
  for frame in tqdm(sorted(frames)):
    label = np.load(os.path.join(directory, f'{frame}_label.npy'))
    if ignore_other:
      label[label >= 37] = 255
    pred = np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
    # handle nans as misclassification
    pred[pred == -1] = 39
    pred[pred == np.nan] = 39
    pred[pred == 255] = 39
    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      torch_label = torch.from_numpy(label)
      valid_pred = torch.from_numpy(pred[torch_label != 255])
      valid_label = torch_label[torch_label != 255]
      cm.update(valid_pred, valid_label)
  cm = cm.compute().numpy().astype(np.uint32)
  return measure_from_confusion_matrix(cm, is_prediction='pred' in method)


def get_measurements(path, ignore_other=True):
  directory = os.path.join(EXP_OUT, path)
  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  methods = list(
      set(
          filename.split(frames[0])[-1].split('.')[0][1:]
          for filename in os.listdir(directory)
          if filename.startswith(frames[0])))
  measurements = {}
  for method in methods:
    frame = sorted(frames)[0]
    pred = np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
    if not np.issubdtype(pred.dtype, np.integer) or pred.dtype == np.uint32:
      print(f'Ignoring {method} because data is not integer type.')
      continue
    measurements[method] = get_measurements_of_method(path,
                                                      method,
                                                      ignore_other=ignore_other)
  return measurements
