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


def get_measurements(path, prediction='pred'):
  directory = os.path.join(EXP_OUT, path)
  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  methods = list(
      set(
          filename.split(frames[0])[-1].split('.')[0][1:]
          for filename in os.listdir(directory)
          if filename.startswith(frames[0])))

  measurements = {}
  ap = torchmetrics.AveragePrecision(compute_on_step=False)
  ap.set_dtype(torch.half)
  auroc = torchmetrics.AUROC(compute_on_step=False)
  auroc.set_dtype(torch.half)
  for method in methods:
    ap.reset()
    ap.set_dtype(torch.half)
    auroc.reset()
    auroc.set_dtype(torch.half)
    for frame in tqdm(frames):
      label = np.load(os.path.join(directory, f'{frame}_label.npy'))
      pred = np.load(os.path.join(directory, f'{frame}_{prediction}.npy'))
      # sort pixels into correct (0), misclassified (1), or ignore (2)
      mc = (label != pred).astype('int')
      mc[label == 255] = 2
      mc[pred == np.nan] = 2
      del label, pred
      if np.sum(mc < 2) == 0:
        continue
      val = -np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
      if not np.issubdtype(val.dtype, np.floating):
        print(f'Ignoring {method} because data is not floating type.')
        methods.remove(method)
        break
      ap.update(torch.from_numpy(-val[mc < 2]), torch.from_numpy(mc[mc < 2]))
      auroc.update(torch.from_numpy(-val[mc < 2]), torch.from_numpy(mc[mc < 2]))
      del mc, val
    if method in methods:
      measurements[method] = {
          'AP': float(ap.compute().numpy()),
          'AUROC': float(auroc.compute().numpy())
      }
  return measurements


@ex.command
def measure(path):
  measurements = get_measurements(path)
  with pd.option_context('display.float_format', '{:0.4f}'.format):
    print(pd.DataFrame.from_dict(measurements).T)


if __name__ == '__main__':
  ex.run_commandline()
