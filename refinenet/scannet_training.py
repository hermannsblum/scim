from sacred import Experiment
import torch
import tensorflow_datasets as tfds
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import make_archive, copyfile

import semseg_density.data.scannet
from semseg_density.data.tfds_to_torch import TFDataIterableDataset
from semseg_density.data.augmentation import augmentation
from semseg_density.data.images import convert_img_to_float
from semseg_density.model.refinenet import rf_lw50, rf_lw101, get_encoder_and_decoder_params
from semseg_density.lr_scheduler import LRScheduler
from semseg_density.segmentation_metrics import SegmentationMetric
from semseg_density.losses import MixSoftmaxCrossEntropyLoss
from semseg_density.settings import TMPDIR
from semseg_density.sacred_utils import get_observer

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
  filename = 'refinenet_scannet.pth'
  save_path = os.path.join(TMPDIR, filename)
  torch.save(model.state_dict(), save_path)
  if postfix is not None:
    best_filename = f'refinenet_scannet_{postfix}.pth'
    best_filename = os.path.join(TMPDIR, best_filename)
    copyfile(save_path, best_filename)


@ex.main
def train(_run,
          batchsize=10,
          epochs=100,
          size=50,
          encoder_lr=5e-4,
          decoder_lr=5e-3,
          subset='25k',
          device='cuda'):
  # DATA LOADING
  data = tfds.load(f'scan_net/{subset}', split='train', as_supervised=True)
  valdata = data.take(1000)
  traindata = data.skip(1000)

  def data_converter(image, label):
    image = convert_img_to_float(image)
    label = tf.cast(label, tf.int64)
    # the output is 4 times smaller than the input, so transform labels
    label = tf.image.resize(label[..., tf.newaxis], (120, 160),
                            method='nearest')[..., 0]
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

  traindata = TFDataIterableDataset(
      traindata.cache().prefetch(10000).map(augmentation).map(data_converter))
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
  if size == 50:
    model = rf_lw50(40, imagenet=True)
  elif size == 101:
    model = rf_lw101(40, imagenet=True)
  else:
    raise UserWarning("Unknown model size.")
  model.to(device)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])

  criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
  encoder_params, decoder_params = get_encoder_and_decoder_params(model)
  encoder_optimizer = torch.optim.Adam(encoder_params, lr=encoder_lr)
  decoder_optimizer = torch.optim.Adam(decoder_params, lr=decoder_lr)
  encoder_lr_scheduler = LRScheduler(mode='poly',
                                     base_lr=encoder_lr,
                                     nepochs=epochs,
                                     iters_per_epoch=len(train_loader),
                                     power=.9)
  decoder_lr_scheduler = LRScheduler(mode='poly',
                                     base_lr=decoder_lr,
                                     nepochs=epochs,
                                     iters_per_epoch=len(train_loader),
                                     power=.9)
  metric = SegmentationMetric(40)

  def validation(epoch, best_pred):
    is_best = False
    metric.reset()
    model.eval()
    for i, (image, target) in enumerate(val_loader):
      image = image.to(device)
      outputs = model(image)
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
      current_encoder_lr = encoder_lr_scheduler(cur_iters)
      for param_group in encoder_optimizer.param_groups:
        param_group['lr'] = current_encoder_lr
      current_decoder_lr = decoder_lr_scheduler(cur_iters)
      for param_group in decoder_optimizer.param_groups:
        param_group['lr'] = current_decoder_lr

      images = images.to(device)
      targets = targets.to(device)

      outputs = model(images)
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
    _run.log_scalar('encoder_lr', current_encoder_lr, epoch)
    _run.log_scalar('decoder_lr', current_decoder_lr, epoch)
    validation(epoch, best_pred)
    if epoch % 5 == 0:
      save_checkpoint(model, postfix=f'{epoch}epochs')
      _run.add_artifact(
          os.path.join(TMPDIR, f'refinenet_scannet_{epoch}epochs.pth'))

  save_checkpoint(model)

  # upload checkpoints
  for filename in ('refinenet_scannet.pth', 'refinenet_scannet_best.pth'):
    modelpath = os.path.join(TMPDIR, filename)
    _run.add_artifact(modelpath)
  return best_pred


if __name__ == '__main__':
  ex.run_commandline()
