from sacred import Experiment
import os
import pandas as pd
import tensorflow_datasets as tfds
import torch
import numpy as np
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Memory

from deeplab.oaisys_utils import OAISYS_LABELS, OAISYS_LABELS_SHORT
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.data.nyu_depth_v2 import TRAINING_LABEL_NAMES

memory = Memory(EXP_OUT)

ex = Experiment()

@memory.cache
def get_measurement(pretrained_id, subset, method, out_classes, only_newclass=False):
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  data = tfds.load(f'{subset}', split='validation')
  # map labels to in-domain (0) or out-domain (1)
  ood_map = 255 * np.ones(256, dtype='uint8')
  ood_map[:12] = 0
  for c in range(12):
    if OAISYS_LABELS_SHORT[c] in out_classes:
      ood_map[c] = 1
  ap = torchmetrics.AveragePrecision(compute_on_step=False)
  ap.set_dtype(torch.half)
  auroc = torchmetrics.AUROC(compute_on_step=False)
  auroc.set_dtype(torch.half)
  roc = torchmetrics.ROC(compute_on_step=False)
  roc.set_dtype(torch.half)
  ap.reset()
  ap.set_dtype(torch.half)
  auroc.reset()
  auroc.set_dtype(torch.half)
  roc.reset()
  roc.set_dtype(torch.half)
  for idx in tqdm(range(len(data))):
    try: 
      label = np.load(os.path.join(directory, f'{idx}_label.npy'))
    except FileNotFoundError:
      try:
        label = np.load(os.path.join(directory, f'{idx}_ensemble_label.npy'))
      except FileNotFoundError:
        print(f'Could not find label for {idx}')
        break
    if only_newclass and not np.any(label == 15):
      continue
    label = np.where(label == 15, 11, label)
    # print(np.unique(label))
    ood = ood_map[label].squeeze()
    if np.sum(ood < 2) == 0:
      continue
    if os.path.exists(os.path.join(directory, f'{idx}_{method}.npy')):
      val = np.load(os.path.join(directory, f'{idx}_{method}.npy')).squeeze()
    else:
      try:
        val = np.load(os.path.join(directory, f'{idx:06d}_{method}.npy')).squeeze()
      except FileNotFoundError:
        return {}
    if 'gmm' in method:
      val *= -1
    if not np.issubdtype(val.dtype, np.floating):
      print(f'Ignoring {method} because data is not floating type.')
      return None
    ap.update(torch.from_numpy(val[ood < 2]), torch.from_numpy(ood[ood < 2]))
    auroc.update(torch.from_numpy(val[ood < 2]),
                 torch.from_numpy(ood[ood < 2]))
    roc.update(torch.from_numpy(val[ood < 2]), torch.from_numpy(ood[ood < 2]))
  fpr, tpr, thresholds = roc.compute()
  ap_val = float(ap.compute().numpy())
  auroc_val = float(auroc.compute().numpy())
  ap.reset()
  auroc.reset()
  roc.reset()

  spec = 1 - fpr
  g_mean = np.sqrt(tpr * spec)
  ideal_idx = np.argmax(g_mean)
  ideal_treshold = thresholds[ideal_idx]

  cm = torchmetrics.ConfusionMatrix(compute_on_step=False, threshold=ideal_treshold, num_classes=2)
  for idx in tqdm(range(len(data))):
    try: 
      label = np.load(os.path.join(directory, f'{idx}_label.npy'))
    except FileNotFoundError:
      try:
        label = np.load(os.path.join(directory, f'{idx}_ensemble_label.npy'))
      except FileNotFoundError:
        print(f'Could not find label for {idx}')
        break
    if only_newclass and not np.any(label == 15):
      continue
    label = np.where(label == 15, 11, label)
    # print(np.unique(label))
    ood = ood_map[label].squeeze()
    if np.sum(ood < 2) == 0:
      continue
    if os.path.exists(os.path.join(directory, f'{idx}_{method}.npy')):
      val = np.load(os.path.join(directory, f'{idx}_{method}.npy')).squeeze()
    else:
      try:
        val = np.load(os.path.join(directory, f'{idx:06d}_{method}.npy')).squeeze()
      except FileNotFoundError:
        return {}
    if 'gmm' in method:
      val *= -1
    if not np.issubdtype(val.dtype, np.floating):
      print(f'Ignoring {method} because data is not floating type.')
      return None
    cm.update(torch.from_numpy(val[ood < 2]), torch.from_numpy(ood[ood < 2]))
  
  tn, fp, fn, tp = cm.compute().numpy().ravel()

  return {
    'AP': ap_val,
    'AUROC': auroc_val,
    'FPR': fpr.numpy(),
    'TPR': tpr.numpy(),
    'Threshold': thresholds.numpy(),
    'G-Mean': g_mean,
    'Ideal_Threshold': ideal_treshold,
    'TP': tp,
    'FP': fp,
    'TN': tn,
    'FN': fn,
    'Precision': tp / (tp + fp),
    'Recall': tp / (tp + fn),
    'F1': 2 * tp / (2 * tp + fp + fn),
    'Accuracy': (tp + tn) / (tp + tn + fp + fn),
  }


def get_measurements(subsets, pretrained_id, out_classes=['Sand'], methods=None, only_newclass=False):
  measurements = {}
  for subset in subsets:
    measurement = {}
    for method in methods:
      measurement[method] = get_measurement(pretrained_id, subset, method, out_classes, only_newclass)
      # measurements[subset][method] = get_measurement(pretrained_id, subset, method, out_classes, only_newclass)
    measurements[subset] = measurement
  return measurements


@ex.command
def measure(path, out_classes):
  measurements = get_measurements(path, out_classes)
  with pd.option_context('display.float_format', '{:0.4f}'.format):
    print(pd.DataFrame.from_dict(measurements).T, flush=True)


if __name__ == '__main__':
  ex.run_commandline()
