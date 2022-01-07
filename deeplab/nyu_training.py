from sacred import Experiment
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile

import semseg_density.data.nyu_depth_v2
from semseg_density.data.tfds_to_torch import TFDataIterableDataset
from semseg_density.data.augmentation import augmentation
from semseg_density.data.images import convert_img_to_float
from semseg_density.model.refinenet import rf_lw50, rf_lw101, get_encoder_and_decoder_params
from semseg_density.lr_scheduler import LRScheduler
from semseg_density.segmentation_metrics import SegmentationMetric
from semseg_density.losses import MixSoftmaxCrossEntropyLoss, MaxLogitLoss
from semseg_density.settings import TMPDIR
from semseg_density.sacred_utils import get_observer, get_checkpoint

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
    model.load_state_dict(new_state_dict, strict=strict)
  else:
    model.load_state_dict(state_dict, strict=strict)


def save_checkpoint(model, postfix=None):
  """Save Checkpoint"""
  filename = 'deeplab_nyu.pth'
  save_path = os.path.join(TMPDIR, filename)
  if hasattr(model, 'module'):
    torch.save(model.module.state_dict(), save_path)
  else:
    torch.save(model.state_dict(), save_path)
  if postfix is not None:
    best_filename = f'deeplab_nyu_{postfix}.pth'
    best_filename = os.path.join(TMPDIR, best_filename)
    copyfile(save_path, best_filename)


ex.add_config(batchsize=10,
              epochs=100,
              lr=0.0001,
              start='imagenet',
              ignore_other=True,
              subset='labeled',
              device='cuda')


@ex.main
def deeplab_nyu(_run, batchsize, epochs, lr, start, ignore_other, subset,
                device):
  # DATA LOADING
  data = tfds.load(f'nyu_depth_v2_labeled/{subset}',
                   split='train',
                   as_supervised=True)
  valdata = data.take(500)
  traindata = data.skip(500)

  def data_converter(image, label):
    image = convert_img_to_float(image)
    label = tf.cast(label, tf.int64)
    if ignore_other:
      label = tf.where(label >= 37, tf.cast(255, tf.int64), label)
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

  traindata = TFDataIterableDataset(
      traindata.cache().prefetch(10000).map(lambda x, y: augmentation(
          x, y, random_crop=(256, 256))).map(data_converter))
  valdata = TFDataIterableDataset(valdata.map(data_converter))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)
  val_loader = torch.utils.data.DataLoader(dataset=valdata,
                                           batch_size=batchsize,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  if start == 'imagenet':
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        pretrained_backbone=True,
        progress=True,
        num_classes=40,
        aux_loss=None)

  else:
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        pretrained_backbone=False,
        progress=True,
        num_classes=40,
        aux_loss=None)
    load_checkpoint(model, get_checkpoint(start)[0])
  model.to(device)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])

  criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_scheduler = LRScheduler(mode='poly',
                             base_lr=lr,
                             nepochs=epochs,
                             iters_per_epoch=len(train_loader),
                             offset=len(train_loader) * 50,
                             power=.9)
  metric = SegmentationMetric(40)

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
            (epoch, epochs, i + 1, len(train_loader), time.time() - start_time,
             loss.item()),
            flush=True)
    _run.log_scalar('loss', loss.item(), epoch)
    _run.log_scalar('lr', current_lr, epoch)
    with torch.no_grad():
      validation(epoch, best_pred)
    if epoch % 5 == 0:
      save_checkpoint(model, postfix=f'{epoch:05d}epochs')
      _run.add_artifact(
          os.path.join(TMPDIR, f'deeplab_nyu_{epoch:05d}epochs.pth'))

  save_checkpoint(model)

  # upload checkpoints
  for filename in ('deeplab_nyu.pth', 'deeplab_nyu_best.pth'):
    modelpath = os.path.join(TMPDIR, filename)
    _run.add_artifact(modelpath)
  return best_pred


if __name__ == '__main__':
  ex.run_commandline()
