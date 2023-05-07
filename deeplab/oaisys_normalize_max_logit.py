from sacred import Experiment
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)

from semsegcluster.data.tfds_to_torch import TFDataIterableDataset
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_checkpoint

ex = Experiment()

def load_checkpoint(model, state_dict, strict=True):
  """Load Checkpoint from Google Drive."""
  # if we currently don't use DataParallel, we have to remove the 'module' prefix
  # from all weight keys
  if (not next(iter(model.state_dict())).startswith('module')) and (next(
      iter(state_dict)).startswith('module')):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict, strict=strict)
  else:
    model.load_state_dict(state_dict, strict=strict)


ex.add_config(
    dataset='oaisys16k_rugd',
    device='cuda',
    use_euler=False,
    num_classes=15,
    training_set='oaisys16k_rugd',
)

def data_converter_rugd(image, label):
  image = convert_img_to_float(image)
  label = tf.squeeze(tf.cast(label, tf.int64))

  label = tf.where(label == 0, tf.cast(255, tf.int64), label)
  label = tf.where(label == 1, tf.cast(0, tf.int64), label)
  label = tf.where(label == 2, tf.cast(15, tf.int64), label)
  label = tf.where(label == 3, tf.cast(1, tf.int64), label)
  label = tf.where(label == 4, tf.cast(2, tf.int64), label)
  label = tf.where(label == 5, tf.cast(3, tf.int64), label)
  label = tf.where(label == 6, tf.cast(4, tf.int64), label)
  label = tf.where(label == 7, tf.cast(5, tf.int64), label)
  label = tf.where(label == 8, tf.cast(3, tf.int64), label)
  label = tf.where(label == 9, tf.cast(3, tf.int64), label)
  label = tf.where(label == 10, tf.cast(255, tf.int64), label)
  label = tf.where(label == 11, tf.cast(6, tf.int64), label)
  label = tf.where(label == 12, tf.cast(255, tf.int64), label)
  label = tf.where(label == 13, tf.cast(7, tf.int64), label)
  label = tf.where(label == 14, tf.cast(8, tf.int64), label)
  label = tf.where(label == 15, tf.cast(9, tf.int64), label)
  label = tf.where(label == 16, tf.cast(3, tf.int64), label)
  label = tf.where(label == 17, tf.cast(3, tf.int64), label)
  label = tf.where(label == 18, tf.cast(3, tf.int64), label)
  label = tf.where(label == 19, tf.cast(2, tf.int64), label)
  label = tf.where(label == 20, tf.cast(3, tf.int64), label)
  label = tf.where(label == 21, tf.cast(10, tf.int64), label)
  label = tf.where(label == 22, tf.cast(255, tf.int64), label)
  label = tf.where(label == 23, tf.cast(255, tf.int64), label)
  label = tf.where(label == 24, tf.cast(3, tf.int64), label)
  label = tf.where(label >= 15, tf.cast(1, tf.int64), label)  # grass label where labeled wrong by oaisys postprocessing

  # move channel from last to 2nd
  image = tf.transpose(image, perm=[2, 0, 1])
  return image, label


@ex.main
def normailize_max_logit_oaisys(dataset, pretrained_model, device, num_classes, use_euler, training_set):
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{dataset}.tar')
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
  else:
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True)


  traindata = TFDataIterableDataset(traindata.map(data_converter_rugd))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=1,
                                             pin_memory=True,
                                             drop_last=True)

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=num_classes,
      aux_loss=None)
  checkpoint, pretrained_id = get_checkpoint(pretrained_model)
  # remove any aux classifier stuff
  removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
  for k in removekeys:
    del checkpoint[k]
  load_checkpoint(model, checkpoint)

  training_directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{training_set}', pretrained_id)
  num_labels = np.load(os.path.join(training_directory, 'num_labels.npy'))
  
  model.to(device)
  model.eval()

  # make sure the directory exists
  directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{dataset}', pretrained_id)
  os.makedirs(directory, exist_ok=True)
  num_labels=np.zeros(num_classes, dtype=int)
  sum_max_logit=np.zeros(num_classes)
  for idx, (image, label) in tqdm(enumerate(train_loader)):
    image = image.to(device)
    # run inference
    logits = model(image)['out']
    max_logit, pred = torch.max(logits, 1)
    cpu_pred = pred.to('cpu')
    cpu_max_logit = max_logit.to('cpu')
    for i in range(0, num_classes):
      num_labels[i]+=torch.sum(torch.where(cpu_pred==i,1,0))
      sum_max_logit[i]+=torch.sum(torch.where(cpu_pred==i,cpu_max_logit, torch.tensor(0, dtype=torch.float32).to('cpu')))
  mean_max_logit=np.zeros(num_classes)
  for idx in range(num_classes):
    if num_labels[idx]!=0:
      mean_max_logit[idx]=sum_max_logit[idx]/num_labels[idx]
  # store outputs
  np.save(os.path.join(directory, f'num_labels.npy'),
          num_labels)
  np.save(os.path.join(directory, f'mean_max_logit.npy'),
          mean_max_logit)
  sum_max_logit_dist=np.zeros(num_classes)
  for idx, (image, label) in tqdm(enumerate(train_loader)):
    image = image.to(device)
    # run inference
    logits = model(image)['out']
    max_logit, pred = torch.max(logits, 1)
    cpu_pred = pred.to('cpu')
    cpu_max_logit = max_logit.to('cpu')
    for i in range(0, num_classes):
      sum_max_logit_dist[i]+=torch.sum(torch.where(cpu_pred==i,(cpu_max_logit-mean_max_logit[i])*(cpu_max_logit-mean_max_logit[i]), torch.tensor(0, dtype=torch.float32).to('cpu')))
  sigma_max_logit=np.zeros(num_classes)
  for idx in range(num_classes):
    if num_labels[idx]!=0:
      sigma_max_logit[idx]=sum_max_logit_dist[idx]/num_labels[idx]
  # store outputs
  np.save(os.path.join(directory, f'sigma_max_logit.npy'),
          sigma_max_logit)





if __name__ == '__main__':
  ex.run_commandline()
