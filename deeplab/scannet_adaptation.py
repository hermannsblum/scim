from sacred import Experiment
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile

import semsegcluster.data.scannet
from semsegcluster.data.nyu_depth_v2 import TRAINING_LABEL_NAMES
from semsegcluster.data.tfds_to_torch import TFDataIterableDataset
from semsegcluster.data.augmentation import augmentation
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.lr_scheduler import LRScheduler
from semsegcluster.segmentation_metrics import SegmentationMetric
from semsegcluster.losses import MixSoftmaxCrossEntropyLoss
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_observer, get_checkpoint

ex = Experiment()
ex.observers.append(get_observer())


def load_checkpoint(model, state_dict, strict=True):
  """Load Checkpoint from Google Drive."""
  # if we currently don't use DataParallel, we have to remove the 'module' prefix
  # from all weight keys
  if (not next(iter(model.state_dict())).startswith('module')) and (next(
      iter(state_dict)).startswith('module')):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      new_state_dict[k[7:]] = v
  else:
    new_state_dict = state_dict
  for k, v in new_state_dict.items():
    if k in [
        'classifier.4.bias', 'classifier.4.weight', 'aux_classifier.4.bias',
        'aux_classifier.4.weight'
    ]:
      random_matrix = model.state_dict()[k]
      if len(v.shape) == 1:
        v = torch.nn.functional.pad(v, (0, 10))
      if len(v.shape) == 4:
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 0, 0, 10))
        assert random_matrix[-10:].shape == (10, 256, 1, 1)
      v[-10:] = random_matrix[-10:]
      new_state_dict[k] = v
  model.load_state_dict(new_state_dict, strict=strict)


def save_checkpoint(model, postfix=None):
  """Save Checkpoint"""
  filename = 'deeplab_scannet.pth'
  save_path = os.path.join(TMPDIR, filename)
  if hasattr(model, 'module'):
    torch.save(model.module.state_dict(), save_path)
  else:
    torch.save(model.state_dict(), save_path)
  if postfix is not None:
    best_filename = f'deeplab_scannet_{postfix}.pth'
    best_filename = os.path.join(TMPDIR, best_filename)
    copyfile(save_path, best_filename)


ex.add_config(
    batchsize=10,
    epochs=30,
    lr=1e-5,
    ignore_other=True,
    keep_threshold=1e4,
    uhlemeyer_only_single_class=True,
)


@ex.main
def train(
    _run,
    batchsize,
    epochs,
    pretrained_model,
    lr,
    ignore_other,
    aux_loss,
    subset,
    pseudolabels,
    keep_threshold,
    uhlemeyer_only_single_class,
    device='cuda',
):
  # DATA LOADING
  data = tfds.load(f'scan_net/{subset}', split='validation')
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset,
                           str(pretrained_model))
  # check if labels are PNG or NPY
  first_frame = next(iter(data))['name'].numpy().decode()
  is_png = os.path.exists(
      os.path.join(directory, f'{first_frame}_{pseudolabels}.png'))

  # filter labels > 40 for those that actually exist
  label_counts = np.zeros(1, dtype=np.int64)
  for blob in data:
    frame = blob['name'].numpy().decode()
    if is_png:
      label = cv2.imread(os.path.join(directory, f'{frame}_{pseudolabels}.png'),
                         cv2.IMREAD_ANYDEPTH).squeeze().astype(np.int32)
      label -= 1
    else:
      label = np.load(os.path.join(directory,
                                   f'{frame}_{pseudolabels}.npy')).squeeze()
    count = np.bincount(label[label >= 40])
    if count.shape[0] > label_counts.shape[0]:
      label_counts = np.pad(label_counts,
                            ((0, count.shape[0] - label_counts.shape[0])))
    else:
      count = np.pad(count, ((0, label_counts.shape[0] - count.shape[0])))
    label_counts += count
  label_map = np.arange(label_counts.shape[0] + 1)
  min_count = max(keep_threshold,
                  np.partition(label_counts.flatten(), -10)[-10])
  new_cluster = 40
  for i in range(40, label_counts.shape[0]):
    if label_counts[i] < min_count or i == 255 or (
        uhlemeyer_only_single_class and 'uhlemeyer' in pseudolabels and
        label_counts[i] != label_counts.max()):
      label_map[i] = 255
    else:
      print(f'keeping cluster {i} with {label_counts[i]} pixels')
      label_map[i] = new_cluster
      new_cluster += 1
  # label -1 gets mapped to 255
  label_map[-1] = 255

  def data_generator():
    for blob in data:
      image = convert_img_to_float(blob['image'])
      frame = blob['name'].numpy().decode()
      if is_png:
        label = cv2.imread(
            os.path.join(directory, f'{frame}_{pseudolabels}.png'),
            cv2.IMREAD_ANYDEPTH).squeeze().astype(np.int32)
        label -= 1
      else:
        label = np.load(os.path.join(directory,
                                     f'{frame}_{pseudolabels}.npy')).squeeze()
      label = label_map[label]
      yield image, label

  gendata = tf.data.Dataset.from_generator(
      data_generator,
      output_signature=(tf.TensorSpec(shape=(480, 640, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(480, 640),
                                      dtype=tf.int32))).shuffle(1000)
  valdata = gendata.take(200)
  traindata = gendata.skip(200)

  def data_converter(image, label):
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    label = tf.cast(label, tf.int64)
    return image, label

  if torch.cuda.device_count() > 1:
    traindata = TFDataIterableDataset(
        traindata.cache().prefetch(10000).map(lambda x, y: augmentation(
            x, y, random_crop=(256, 256))).map(data_converter))
  else:
    traindata = TFDataIterableDataset(
        traindata.map(lambda x, y: augmentation(x, y, random_crop=(256, 256))).
        map(data_converter))
  valdata = TFDataIterableDataset(valdata.map(data_converter))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)
  val_loader = torch.utils.data.DataLoader(dataset=valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=50,
      aux_loss=aux_loss)
  load_checkpoint(model, get_checkpoint(pretrained_model)[0])
  model.to(device)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])

  criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_scheduler = LRScheduler(mode='poly',
                             base_lr=lr,
                             nepochs=epochs,
                             iters_per_epoch=len(data),
                             power=.9)
  metric = SegmentationMetric(50)

  def validation(epoch, best_pred):
    is_best = False
    metric.reset()
    model.eval()
    for i, (image, target) in enumerate(val_loader):
      image = image.to(device)
      outputs = model(image)['out']
      pred = torch.argmax(outputs, 1)
      pred = pred.cpu().data.numpy()
      metric.update(pred, target.numpy())
    pixAcc, mIoU = metric.get()
    print('Epoch %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' %
          (epoch, pixAcc * 100, mIoU * 100))
    _run.log_scalar('val_miou', mIoU, epoch)
    _run.log_scalar('val_acc', pixAcc, epoch)

    new_pred = (pixAcc + mIoU) / 2
    if new_pred > best_pred:
      is_best = True
      best_pred = new_pred
    save_checkpoint(model, postfix='best' if is_best else None)

  best_pred = .0
  cur_iters = 0
  start_time = time.time()
  for epoch in range(epochs):
    model.train()

    for i, (images, targets) in enumerate(train_loader):
      # learning-rate update
      current_lr = lr_scheduler(cur_iters)
      for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

      images = images.to(device)
      targets = targets.to(device)

      outputs = model(images)['out']
      loss = criterion(outputs, targets)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      cur_iters += 1
      if cur_iters % 100 == 0:
        print(
            'Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || Loss: %.4f' %
            (epoch, epochs, i + 1, (len(data) - 200) // batchsize,
             time.time() - start_time, loss.item()),
            flush=True)
    _run.log_scalar('loss', loss.item(), epoch)
    _run.log_scalar('lr', current_lr, epoch)
    del images, targets, outputs, loss
    with torch.no_grad():
      validation(epoch, best_pred)
    if epoch % 5 == 0:
      save_checkpoint(model, postfix=f'{epoch}epochs')

  save_checkpoint(model)

  # upload checkpoint
  _run.add_artifact(os.path.join(TMPDIR, 'deeplab_scannet_best.pth'))
  time.sleep(5)
  return best_pred


if __name__ == '__main__':
  ex.run_commandline()
