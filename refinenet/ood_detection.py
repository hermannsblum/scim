from sacred import Experiment
import os
import torch
import numpy as np
import torchmetrics
from tqdm import tqdm
import matplotlib.pyplot as plt

from semseg_density.gdrive import load_gdrive_file
from semseg_density.settings import TMPDIR, EXP_OUT
from semseg_density.data.nyu_depth_v2 import TRAINING_LABEL_NAMES

ex = Experiment()


@ex.command
def measure(path, out_classes=['pilow', 'refridgerator', 'television']):
  directory = os.path.join(EXP_OUT, path)
  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  methods = set(
      filename.split(frames[0])[-1].split('.')[0][1:]
      for filename in os.listdir(directory)
      if filename.startswith(frames[0]))
  methods.pop('label')
  print(methods)
  method_ap = {
      m: torchmetrics.AveragePrecision(compute_on_step=False) for m in methods
  }

  # map labels to in-domain (0) or out-domain (1)
  ood_map = 255 * np.ones(256, dtype='uint8')
  ood_map[:40] = 0
  for c in range(40):
    if TRAINING_LABEL_NAMES[c] in out_classes:
      ood_map[c] = 1

  for frame in tqdm(frames):
    label = np.load(os.path.join(directory, f'{frame}_label.npy'))
    ood = ood_map[label].squeeze()
    for method in methods:
      val = np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
      method_ap[method].update(torch.from_numpy(val), torch.from_numpy(ood))

  for method in methods:
    print(f'{method}: {method_ap[method].compute() * 100:.2f}% AP')


@ex.main
def scatter(
    path,
    a='nll',
    b='maxlogit',
    out_classes=['pilow', 'refridgerator', 'television'],
):
  # make sure the directory exists, but is empty
  directory = os.path.join(EXP_OUT, path)

  in_a = []
  out_a = []
  in_b = []
  out_b = []

  # map labels to in-domain (0) or out-domain (1)
  ood_map = 255 * np.ones(256, dtype='uint8')
  ood_map[:40] = 0
  for c in range(40):
    if TRAINING_LABEL_NAMES[c] in out_classes:
      ood_map[c] = 1

  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  for frame in tqdm(frames[:500]):
    label = np.load(os.path.join(directory, f'{frame}_label.npy'))
    ood = ood_map[label].squeeze()
    a_val = np.load(os.path.join(directory, f'{frame}_{a}.npy')).squeeze()
    b_val = np.load(os.path.join(directory, f'{frame}_{b}.npy')).squeeze()
    in_a.append(a_val[ood == 0].reshape((-1)))
    in_b.append(b_val[ood == 0].reshape((-1)))
    out_a.append(a_val[ood == 1].reshape((-1)))
    out_b.append(b_val[ood == 1].reshape((-1)))
  in_a = np.concatenate(in_a)
  out_a = np.concatenate(out_a)
  in_b = np.concatenate(in_b)
  out_b = np.concatenate(out_b)

  plt.figure(figsize=(10, 10))
  plt.scatter(in_a,
              in_b,
              c="black",
              alpha=0.01,
              linewidths=0.0,
              rasterized=True)
  plt.scatter(out_a,
              out_b,
              c="red",
              alpha=0.01,
              linewidths=0.0,
              rasterized=True)
  plt.xlabel(a)
  plt.ylabel(b)
  plt.savefig(os.path.join(directory, f'ood_{a}_{b}_scatter.pdf'), dpi=400)


if __name__ == '__main__':
  ex.run_commandline()
