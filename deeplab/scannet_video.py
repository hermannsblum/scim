from sacred import Experiment
import torch
import torchvision
import torchmetrics
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import cv2
import detectron2 as det2
import detectron2.utils.visualizer
import shutil
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess

tf.config.set_visible_devices([], 'GPU')

import semsegcluster.data.scannet
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.data.nyu_depth_v2 import TRAINING_LABEL_NAMES, NYU40_COLORS
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_incense_loader, get_checkpoint
from semsegcluster.model.postprocessing import BoundarySuppressionWithSmoothing

ex = Experiment()


@ex.main
def videoframes(pretrained_id, subset, prediction):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  m = det2.data.Metadata()
  m.stuff_classes = ['' for _ in range(200)]
  colors = np.stack(NYU40_COLORS + [
      det2.utils.colormap.random_color(rgb=True).astype(np.uint8)
      for _ in range(5000)
  ],
                    axis=0)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset,
                           str(pretrained_id))
  os.makedirs(os.path.join('/tmp', f'rendered-{prediction}'), exist_ok=True)
  for blob in tqdm(data):
    image = blob['image'].numpy()
    frame = blob['name'].numpy().decode()
    v = np.load(os.path.join(directory, f'{frame}_{prediction}.npy')).squeeze()
    #vis = det2.utils.visualizer.Visualizer(image, m, scale=0.4)
    #out = vis.draw_sem_seg(v).get_image()
    pred = colors[v]
    out = 0.8 * pred + 0.2 * image
    out = np.maximum(0, out)
    out = np.minimum(255, out)
    cv2.imwrite(os.path.join('/tmp', f'rendered-{prediction}', f'{frame}.jpg'),
                out[..., ::-1])
  subprocess.call([
      'ffmpeg', '-i', f'/tmp/rendered-{prediction}/{subset}_%06d.jpg', '-c:v',
      'libx264', '-crf', '23', '-vf', 'fps=30',
      f'/home/blumh/ASL/random/{subset}-{prediction}.mp4'
  ])


if __name__ == '__main__':
  ex.run_commandline()
