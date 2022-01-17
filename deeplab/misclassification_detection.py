from sacred import Experiment
import os
import pandas as pd
import torch
import numpy as np
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt

from semseg_density.settings import TMPDIR, EXP_OUT
from semseg_density.data.nyu_depth_v2 import TRAINING_LABEL_NAMES

ex = Experiment()


@ex.command
def measure(path):
  directory = os.path.join(EXP_OUT, path)
  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  methods = list(
      set(
          filename.split(frames[0])[-1].split('.')[0][1:]
          for filename in os.listdir(directory)
          if filename.startswith(frames[0])))
  methods.remove('label')
  methods.remove('pred')
  methods.remove('instances')
  method_ap = {
      m: torchmetrics.AveragePrecision(compute_on_step=False) for m in methods
  }
  method_ap['gt'] = torchmetrics.AveragePrecision(compute_on_step=False)
  method_auroc = {m: torchmetrics.AUROC(compute_on_step=False) for m in methods}
  method_auroc['gt'] = torchmetrics.AUROC(compute_on_step=False)

  for frame in tqdm(frames):
    label = np.load(os.path.join(directory, f'{frame}_label.npy'))
    pred = np.load(os.path.join(directory, f'{frame}_pred.npy'))
    # sort pixels into correct (0), misclassified (1), or ignore (2)
    mc = (label != pred).astype('int')
    mc[label == 255] = 2
    if np.sum(mc < 2) == 0:
      continue
    for method in methods:
      val = -np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
      method_ap[method].update(torch.from_numpy(-val[mc < 2]),
                               torch.from_numpy(mc[mc < 2]))
      method_auroc[method].update(torch.from_numpy(-val[mc < 2]),
                                  torch.from_numpy(mc[mc < 2]))
    method_ap['gt'].update(torch.from_numpy(mc[mc < 2].astype(float)),
                           torch.from_numpy(mc[mc < 2]))
    method_auroc['gt'].update(torch.from_numpy(mc[mc < 2].astype(float)),
                              torch.from_numpy(mc[mc < 2]))

  measurements = {}
  for method in method_ap:
    measurements[method] = {
        'AP': float(method_ap[method].compute().numpy()),
        'AUROC': float(method_auroc[method].compute().numpy())
    }
  with pd.option_context('display.float_format', '{:0.4f}'.format):
    print(pd.DataFrame.from_dict(measurements).T)




if __name__ == '__main__':
  ex.run_commandline()
