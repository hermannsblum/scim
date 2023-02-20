import tensorflow_datasets as tfds
from sacred import Experiment
import os
import numpy as np
from tqdm import tqdm
import cv2

from semsegcluster.settings import EXP_OUT

ex = Experiment()

ex.add_config(
    uncertainty_threshold=-3,
)


@ex.main
def create_pseudolabel(subset, uncertainty_threshold):
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, 'deeplab')
  data = tfds.load(f'{subset}', split='validation')
  for idx in tqdm(range(len(data))):
    u = np.load(os.path.join(directory,
                                   f'{idx}_maxlogit-pp.npy')).squeeze()
    i = np.load(os.path.join(directory, f'{idx}_pred.npy')).squeeze()
    o = np.load(os.path.join(directory,
                             f'{idx}_label.npy')).squeeze()
    train_signal = i
    train_signal[u > uncertainty_threshold] = o[u > uncertainty_threshold]
    train_signal = train_signal.astype(np.uint16)
    if train_signal.max() < 255:
      train_signal = train_signal.astype(np.uint8)
    cv2.imwrite(
        os.path.join(directory, f'{idx}_merged-pred-label.png'),
        train_signal)


if __name__ == '__main__':
  ex.run_commandline()
