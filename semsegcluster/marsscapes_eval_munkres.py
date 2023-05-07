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
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from semsegcluster.settings import EXP_OUT
from semsegcluster.segmentation_metrics import SegmentationMetric
from deeplab.oaisys_utils import OAISYS_LABELS

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
  }
  # contingency matrix based sklearn metrics
  # taken from https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/metrics/cluster/_supervised.py#L402
  cm = cm.astype(np.float64)
  n_total = cm.sum()
  n_labels = cm.sum(1)
  n_labels = n_labels[n_labels > 0]
  entropy_labels = -np.sum(
      (n_labels / n_total) * (np.log(n_labels) - np.log(n_total)))
  n_pred = cm.sum(0)
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
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  cm = torchmetrics.ConfusionMatrix(num_classes=200)
  data = tfds.load(f'{subset}', split='validation')
  is_prediction = (('merged' not in method and 'pred' in method) or ('merged' in method and 'pseudolabel' in method))
  for idx in tqdm(range(len(data))):
    label = np.load(os.path.join(directory, f'{idx}_label.npy')).squeeze()
    if ignore_other:
      label[label >= 37] = 255
    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      label = torch.from_numpy(label)
      if os.path.exists(os.path.join(directory, f'{idx}_{method}.npy')):
        pred = np.load(os.path.join(directory, f'{idx}_{method}.npy')).squeeze()
      else:
        pred = np.load(os.path.join(directory, f'{idx:06d}_{method}.npy')).squeeze()
      # make space for wildcard cluster 0
      pred += 1
      if 'seg' in method:
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
def get_measurements_of_method_with_munkres(subset,
                                            pretrained_id,
                                            method,
                                            ignore_other=True):
  # directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)


  directory = '/home/asl/Downloads/file_transfer/marsscapes/inference/deeplab/'
  adapted_directory = '/home/asl/Downloads/file_transfer/marsscapes/adapted_inference/deeplab/'
  cluster_directory = '/home/asl/Downloads/file_transfer/marsscapes/clustering/'
  path = '/home/asl/Downloads/MarsScapes/processed'


  cm = torchmetrics.ConfusionMatrix(num_classes=200)
  # data = tfds.load(f'{subset}', split='validation')
  is_prediction = (('merged' not in method and 'pred' in method) or ('merged' in method and 'pseudolabel' in method))
  test_path = f'{path}/test/'
  val_path = f'{path}/val/'
  train_path = f'{path}/train/'
  for image_name in os.listdir(cluster_directory):
    image_name = image_name[:-11]
    label = np.load(os.path.join(directory, f'{image_name}_label.npy')).squeeze()
    # label = np.load(os.path.join(directory, f'{idx}_label.npy')).squeeze()
    if ignore_other:
      label[label >= 37] = 255
    # update confusion matrix, only on labelled pixels
    if np.any(label != 255):
      label = torch.from_numpy(label)
      if 'label' in method:
        pred = label
      elif 'ensemble_pred' in method:
        pred = np.load(os.path.join(directory, f'{image_name}_ensemble_pred.npy')).squeeze()
      elif 'pred401-merged-ensemble_pred-seg107' in method:
        pred = cv2.imread(os.path.join(adapted_directory, f'{image_name}_pred401-merged-ensemble_pred-seg107.png'),
                         cv2.IMREAD_ANYDEPTH).squeeze().astype(np.int32)
      elif 'seg440' in method:
        pred = np.load(os.path.join(cluster_directory, f'{image_name}_seg440.npy')).squeeze()
      else:
        raise ValueError(f'Method {method} not found')
      pred = np.where(label==255, 255, pred)
      # pred = cv2.imread(os.path.join(directory, f'{idx:06d}_{method}.png'),
      #                   cv2.IMREAD_ANYDEPTH).squeeze().astype(np.int32)
      pred += 1
      # handle nans as misclassification
      pred[np.isnan(pred)] = 0
      # cluster numbers larger than 200 are ignored in  the confusionm  matrix
      pred[pred > 200] = 0
      pred = torch.from_numpy(pred)[label != 255]
      label = label[label != 255]
      cm.update(pred, label)
  cm = cm.compute().numpy().astype(np.uint32)
  return measure_from_confusion_matrix(cm, is_prediction=is_prediction)


def get_measurements(subset, pretrained_id, methods, ignore_other=True):
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  measurements = {}
  for method in methods:
    measurements[method] = get_measurements_of_method_with_munkres(
          subset, pretrained_id, method, ignore_other=ignore_other)
  return measurements

def get_ood(subset, pretrained_id, methods, uncert_treshold ,out_classes=['sand']):
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)

  # map labels to in-domain (0) or out-domain (1)
  ood_map = 255 * np.ones(256, dtype='uint8')
  ood_map[:11] = 0
  for c in range(11):
    if OAISYS_LABELS[c] in out_classes:
      ood_map[c] = 1

  measurements = {}
  ap = torchmetrics.AveragePrecision(compute_on_step=False)
  ap.set_dtype(torch.half)
  auroc = torchmetrics.AUROC(compute_on_step=False)
  auroc.set_dtype(torch.half)

  data = tfds.load(f'{subset}', split='validation')
  cm = torchmetrics.ConfusionMatrix(num_classes=16, compute_on_step=False)

  for method in methods:
    ap.reset()
    ap.set_dtype(torch.half)
    auroc.reset()
    auroc.set_dtype(torch.half)
    for idx in tqdm(range(len(data))):
      label = np.load(os.path.join(directory, f'{idx}_label.npy'))
      ood = ood_map[label].squeeze()
      if np.sum(ood < 2) == 0:
        continue
      if os.path.exists(os.path.join(directory, f'{idx}_{method}.npy')):
        val = np.load(os.path.join(directory, f'{idx}_{method}.npy')).squeeze()
      else:
        val = np.load(os.path.join(directory, f'{idx:06d}_{method}.npy')).squeeze()
      if 'gmm' in method:
        val = val < uncert_treshold
      else:
        val = val > uncert_treshold
      cm.update(torch.tensor(label), torch.tensor(val+14))
      # if not np.issubdtype(val.dtype, np.floating):
      #   print(f'Ignoring {method} because data is not floating type.')
      #   methods.remove(method)
      #   break
      # ap.update(torch.from_numpy(val[ood < 2]), torch.from_numpy(ood[ood < 2]))
      # auroc.update(torch.from_numpy(val[ood < 2]),
      #              torch.from_numpy(ood[ood < 2]))
    # CONFUSION MATRIX
  cm = cm.compute().numpy()
  np.save(os.path.join(directory, f'ood_detection_{method}_{uncert_treshold}.npy'), cm)
  disp = ConfusionMatrixDisplay(cm / cm.sum(0),
                                display_labels=OAISYS_LABELS)
  plt.figure(figsize=(20, 20))
  disp.plot(ax=plt.gca(), xticks_rotation='vertical', include_values=False)
  plt.savefig(os.path.join(directory, f'ood_detection_{method}_{uncert_treshold}.pdf'))
    # if method in methods:
    #   measurements[method] = {
    #       'AP': float(ap.compute().numpy()),
    #       'AUROC': float(auroc.compute().numpy())
    #   }

  return
