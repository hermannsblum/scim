from sacred import Experiment
import os
import pandas as pd
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
  methods = list(
      set(
          filename.split(frames[0])[-1].split('.')[0][1:]
          for filename in os.listdir(directory)
          if filename.startswith(frames[0])))
  methods.remove('label')
  method_ap = {
      m: torchmetrics.AveragePrecision(compute_on_step=False) for m in methods
  }
  method_ap['gt'] = torchmetrics.AveragePrecision(compute_on_step=False)
  method_auroc = {m: torchmetrics.AUROC(compute_on_step=False) for m in methods}
  method_auroc['gt'] = torchmetrics.AUROC(compute_on_step=False)

  # map labels to in-domain (0) or out-domain (1)
  ood_map = 255 * np.ones(256, dtype='uint8')
  ood_map[:40] = 0
  for c in range(40):
    if TRAINING_LABEL_NAMES[c] in out_classes:
      ood_map[c] = 1

  for frame in tqdm(frames):
    label = np.load(os.path.join(directory, f'{frame}_label.npy'))
    ood = ood_map[label].squeeze()
    if np.sum(ood < 2) == 0:
      continue
    for method in methods:
      val = -np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
      method_ap[method].update(torch.from_numpy(-val[ood < 2]),
                               torch.from_numpy(ood[ood < 2]))
      method_auroc[method].update(torch.from_numpy(-val[ood < 2]),
                                  torch.from_numpy(ood[ood < 2]))
    method_ap['gt'].update(torch.from_numpy(ood[ood < 2].astype(float)),
                           torch.from_numpy(ood[ood < 2]))
    method_auroc['gt'].update(torch.from_numpy(ood[ood < 2].astype(float)),
                              torch.from_numpy(ood[ood < 2]))

  measurements = {}
  for method in method_ap:
    measurements[method] = {
        'AP': float(method_ap[method].compute().numpy()),
        'AUROC': float(method_auroc[method].compute().numpy())
    }
  with pd.option_context('display.float_format', '{:0.4f}'.format):
    print(pd.DataFrame.from_dict(measurements).T)


@ex.command
def hist3d(path,
           a,
           b,
           out_classes=['pilow', 'refridgerator', 'television'],
           nbins=50):
  directory = os.path.join(EXP_OUT, path)
  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  # map labels to in-domain (0) or out-domain (1)
  ood_map = 255 * np.ones(256, dtype='uint8')
  ood_map[:40] = 0
  for c in range(40):
    if TRAINING_LABEL_NAMES[c] in out_classes:
      ood_map[c] = 1

  a_max = 0.0
  a_min = 0.0
  b_max = 0.0
  b_min = 0.0
  # first compute max and min
  for frame in tqdm(frames):
    val = np.load(os.path.join(directory, f'{frame}_{a}.npy')).squeeze()
    a_max = np.maximum(a_max, val.max())
    a_min = np.minimum(a_min, val.min())
    val = np.load(os.path.join(directory, f'{frame}_{b}.npy')).squeeze()
    b_max = np.maximum(b_max, val.max())
    b_min = np.minimum(b_min, val.min())

  # now compute histograms
  a_edges = np.linspace(a_min, a_max, nbins + 1)
  b_edges = np.linspace(b_min, b_max, nbins + 1)
  id_hist = np.zeros((nbins, nbins))
  ood_hist = np.zeros((nbins, nbins))
  for frame in tqdm(frames):
    label = np.load(os.path.join(directory, f'{frame}_label.npy'))
    ood = ood_map[label].squeeze()
    a_val = np.load(os.path.join(directory, f'{frame}_{a}.npy')).squeeze()
    b_val = np.load(os.path.join(directory, f'{frame}_{b}.npy')).squeeze()
    id_hist += np.histogram2d(a_val[ood == 0],
                              b_val[ood == 0],
                              bins=[a_edges, b_edges])[0]
    ood_hist += np.histogram2d(a_val[ood == 1],
                               b_val[ood == 1],
                               bins=[a_edges, b_edges])[0]

  # normalize
  id_hist = id_hist / id_hist.sum()
  ood_hist = ood_hist / ood_hist.sum()

  plt.figure()
  ax = plt.axes(projection='3d')
  x, y = np.meshgrid(a_edges[:-1], b_edges[:-1])
  #ax.plot_wireframe(a_edges[:-1], b_edges[:-1], id_hist, label='inliers')
  ax.view_init(60, 35)
  ax.plot_wireframe(x, y, id_hist, label='inliers', colors=(0, 0, 0, 0.5))
  ax.plot_wireframe(x, y, ood_hist, label='outliers', colors=(1, 0, 0, 0.5))
  #ax.contour3D(x, y, ood_hist.flatten(), label='outliers')
  plt.legend()
  plt.savefig(os.path.join(directory, f'ood_{a}_{b}_hist.pdf'))


@ex.command
def hist(path,
         method,
         out_classes=['pilow', 'refridgerator', 'television'],
         nbins=50):
  directory = os.path.join(EXP_OUT, path)
  frames = [x[:-10] for x in os.listdir(directory) if x.endswith('label.npy')]
  max_val = 0.0
  min_val = 0.0

  # map labels to in-domain (0) or out-domain (1)
  ood_map = 255 * np.ones(256, dtype='uint8')
  ood_map[:40] = 0
  for c in range(40):
    if TRAINING_LABEL_NAMES[c] in out_classes:
      ood_map[c] = 1

  # first compute max and min
  for frame in tqdm(frames):
    val = np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
    max_val = np.maximum(max_val, val.max())
    min_val = np.minimum(min_val, val.min())

  # now compute histograms
  bin_edges = np.linspace(min_val, max_val, nbins + 1)
  id_hist = np.zeros(nbins)
  ood_hist = np.zeros(nbins)
  for frame in tqdm(frames):
    label = np.load(os.path.join(directory, f'{frame}_label.npy'))
    ood = ood_map[label].squeeze()
    val = np.load(os.path.join(directory, f'{frame}_{method}.npy')).squeeze()
    id_hist += np.histogram(val[ood == 0], bins=bin_edges)[0]
    ood_hist += np.histogram(val[ood == 1], bins=bin_edges)[0]

  # normalize
  id_hist = id_hist / id_hist.sum()
  ood_hist = ood_hist / ood_hist.sum()

  plt.figure()
  plt.bar(bin_edges[:-1],
          id_hist,
          width=(max_val - min_val) / nbins,
          alpha=.4,
          label='inliers')
  plt.bar(bin_edges[:-1],
          ood_hist,
          width=(max_val - min_val) / nbins,
          alpha=.4,
          label='outliers')
  plt.legend()
  plt.savefig(os.path.join(directory, f'ood_{method}_hist.pdf'))


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
