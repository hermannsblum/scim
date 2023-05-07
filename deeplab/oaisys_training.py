import cv2
import PIL
from sacred import Experiment
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

tf.config.set_visible_devices([], 'GPU')
import os
import time
from collections import OrderedDict
from shutil import copyfile
import wandb

import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)

from deeplab.oaisys_sampling import get_deeplab_hooks
from gmmtorch.gmm import GaussianMixture
from semsegcluster.data.tfds_to_torch import TFDataIterableDataset
from semsegcluster.data.augmentation import augmentation
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.lr_scheduler import LRScheduler
from semsegcluster.segmentation_metrics import SegmentationMetric
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_observer

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


def save_checkpoint(model, gmm_model, postfix=None):
  """Save Checkpoint"""
  filename = 'deeplab_oaisys_1000_test.pth'
  save_path = os.path.join(TMPDIR, filename)
  if hasattr(model, 'module'):
    torch.save(model.module.state_dict(), save_path)
  else:
    torch.save(model.state_dict(), save_path)
  if postfix is not None:
    best_filename = f'deeplab_oaisys_1000_test_{postfix}.pth'
    best_filename = os.path.join(TMPDIR, best_filename)
    copyfile(save_path, best_filename)


ex.add_config(
    batchsize=10,
    epochs=100,
    lr=0.0001,
    dataset='oaisys16k_rugd',
    device='cuda',
    aux_loss=False,
    separate_eval=False,
    num_classes=11,
    training_set='oaisys16k_rugd',
    weighted_loss=True,
    use_euler=True,
    feature_name='classifier.2',
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
def deeplab_oaisys(_run, batchsize, epochs, lr, dataset, aux_loss, device, separate_eval, num_classes, training_set, weighted_loss, use_euler):
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{dataset}.tar')
    if separate_eval:
      os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/rugd.tar')
      os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/oaisys11k.tar')


  wandb.init(project=f'{dataset}-pretraining')

  # DATA LOADING
  if use_euler:
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
    valdata = tfds.load(dataset,
                        split='validation',
                        as_supervised=True,
                        data_dir=f'{TMPDIR}/datasets')
  else:
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True)
    valdata = tfds.load(dataset,
                        split='validation',
                        as_supervised=True)

  traindata = TFDataIterableDataset(
      traindata.map(lambda x, y: augmentation(
          x, y, random_crop=(256, 256))).map(data_converter_rugd))
  valdata = TFDataIterableDataset(valdata.map(data_converter_rugd))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)
  val_loader = torch.utils.data.DataLoader(dataset=valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)
  if separate_eval:
    if use_euler:
      rugd_valdata = tfds.load('rugd',
                          split='validation',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
      oaisys_valdata = tfds.load('oaisys11k',
                          split='validation',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
    else:
      rugd_valdata = tfds.load('rugd',
                          split='validation',
                          as_supervised=True)
      oaisys_valdata = tfds.load('oaisys11k',
                          split='validation',
                          as_supervised=True)
    rugd_valdata = TFDataIterableDataset(rugd_valdata.map(data_converter_rugd))
    oaisys_valdata = TFDataIterableDataset(oaisys_valdata.map(data_converter_rugd))
    rugd_val_loader = torch.utils.data.DataLoader(dataset=rugd_valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)
    oaisys_val_loader = torch.utils.data.DataLoader(dataset=oaisys_valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=True,
      progress=True,
      num_classes=num_classes,
      aux_loss=aux_loss)
  model.to(device)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])

  if weighted_loss and training_set is not None:
    training_directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{training_set}', 'deeplab')
    num_labels = np.load(os.path.join(training_directory, 'num_labels.npy'))
    weights = torch.zeros(num_classes)
    for i in range(0, num_classes):
      if num_labels[i] != 0:
        weights[i] = 1 / num_labels[i]
    weights *= np.min(num_labels)

  if weighted_loss:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights).to(device)
  else:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_scheduler = LRScheduler(mode='poly',
                             base_lr=lr,
                             nepochs=epochs,
                             iters_per_epoch=len(train_loader),
                             power=.9)
  metric = SegmentationMetric(num_classes)
  wandb.config.update(
    {
      "learning_rate": lr,
      "epochs": epochs,
      "batch_size": batchsize,
    }
  )

  def validation(epoch, best_pred):
    is_best = False
    metric.reset()
    model.eval()
    print("number of valdata: ", len(val_loader))
    for i, (image, target) in enumerate(val_loader):
      image = image.to(device)
      outputs = model(image)['out']
      pred = torch.argmax(outputs, 1)
      pred = pred.cpu().data.numpy()
      metric.update(pred, target.numpy())
    pixAcc, mIoU, IoU = metric.get_result()
    print('Epoch %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' %
          (epoch, pixAcc * 100, mIoU * 100))
    _run.log_scalar('val_miou', mIoU, epoch)
    _run.log_scalar('val_acc', pixAcc, epoch)
    wandb.log({
      'val_miou': mIoU,
      'val_acc': pixAcc,
      'miou_dirt': IoU[0],
      'miou_grass': IoU[1],
      'miou_tree': IoU[2],
      'miou_object': IoU[3],
      'miou_water': IoU[4],
      'miou_sky': IoU[5],
      'miou_gravel': IoU[6],
      'miou_mulch': IoU[7],
      'miou_bed_rock': IoU[8],
      'miou_log': IoU[9],
      'miou_rock': IoU[10],
      'epoch': epoch,
    })

    new_pred = (pixAcc + mIoU) / 2
    if new_pred > best_pred:
      is_best = True
      best_pred = new_pred
    save_checkpoint(model, postfix='best' if is_best else None)

    if separate_eval:
      metric.reset()
      for i, (image, target) in enumerate(rugd_val_loader):
        image = image.to(device)
        outputs = model(image)['out']
        pred = torch.argmax(outputs, 1)
        pred = pred.cpu().data.numpy()
        metric.update(pred, target.numpy())
      pixAcc, mIoU, IoU = metric.get_result()
      wandb.log({
        'rugd_val_miou': mIoU,
        'rugd_val_acc': pixAcc,
        'rugd_miou_dirt': IoU[0],
        'rugd_miou_grass': IoU[1],
        'rugd_miou_tree': IoU[2],
        'rugd_miou_object': IoU[3],
        'rugd_miou_water': IoU[4],
        'rugd_miou_sky': IoU[5],
        'rugd_miou_gravel': IoU[6],
        'rugd_miou_mulch': IoU[7],
        'rugd_miou_bed_rock': IoU[8],
        'rugd_miou_log': IoU[9],
        'rugd_miou_rock': IoU[10],
        'epoch': epoch,
      })
      metric.reset()
      for i, (image, target) in enumerate(oaisys_val_loader):
        image = image.to(device)
        outputs = model(image)['out']
        pred = torch.argmax(outputs, 1)
        pred = pred.cpu().data.numpy()
        metric.update(pred, target.numpy())
      pixAcc, mIoU, IoU = metric.get_result()
      wandb.log({
        'oaisys_val_miou': mIoU,
        'oaisys_val_acc': pixAcc,
        'oaisys_miou_dirt': IoU[0],
        'oaisys_miou_grass': IoU[1],
        'oaisys_miou_tree': IoU[2],
        'oaisys_miou_object': IoU[3],
        'oaisys_miou_water': IoU[4],
        'oaisys_miou_sky': IoU[5],
        'oaisys_miou_gravel': IoU[6],
        'oaisys_miou_mulch': IoU[7],
        'oaisys_miou_bed_rock': IoU[8],
        'oaisys_miou_log': IoU[9],
        'oaisys_miou_rock': IoU[10],
        'epoch': epoch,
      })


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
      wandb.log({'batch': cur_iters, 'epoch': epoch, 'loss': loss.item(), 'lr': current_lr})
    with torch.no_grad():
      validation(epoch, best_pred)
    if epoch % 2 == 0:
      save_checkpoint(model, postfix=f'{epoch:05d}epochs')
      _run.add_artifact(os.path.join(TMPDIR, f'deeplab_oaisys_1000_test_{epoch:05d}epochs.pth'))

  save_checkpoint(model)

  # upload checkpoints
  for filename in ('deeplab_oaisys_1000_test.pth', 'deeplab_oaisys_1000_test_best.pth'):
    modelpath = os.path.join(TMPDIR, filename)
    _run.add_artifact(modelpath)
  time.sleep(5)
  return best_pred


@ex.command
def deeplab_gmm_targets_oaisys(_run, batchsize, epochs, lr, dataset, aux_loss, device, separate_eval, num_classes, training_set, weighted_loss, use_euler, feature_name):
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{dataset}.tar')
    if separate_eval:
      os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/rugd.tar')
      os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/oaisys11k.tar')


  wandb.init(project=f'{dataset}-gmm-pretraining')

  # DATA LOADING
  if use_euler:
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
    valdata = tfds.load(dataset,
                        split='validation',
                        as_supervised=True,
                        data_dir=f'{TMPDIR}/datasets')
  else:
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True)
    valdata = tfds.load(dataset,
                        split='validation',
                        as_supervised=True)

  traindata = TFDataIterableDataset(
      traindata.map(lambda x, y: augmentation(
          x, y, random_crop=(256, 256))).map(data_converter_rugd))
  valdata = TFDataIterableDataset(valdata.map(data_converter_rugd))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)
  val_loader = torch.utils.data.DataLoader(dataset=valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)
  if separate_eval:
    if use_euler:
      rugd_valdata = tfds.load('rugd',
                          split='validation',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
      oaisys_valdata = tfds.load('oaisys11k',
                          split='validation',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
    else:
      rugd_valdata = tfds.load('rugd',
                          split='validation',
                          as_supervised=True)
      oaisys_valdata = tfds.load('oaisys11k',
                          split='validation',
                          as_supervised=True)
    rugd_valdata = TFDataIterableDataset(rugd_valdata.map(data_converter_rugd))
    oaisys_valdata = TFDataIterableDataset(oaisys_valdata.map(data_converter_rugd))
    rugd_val_loader = torch.utils.data.DataLoader(dataset=rugd_valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)
    oaisys_val_loader = torch.utils.data.DataLoader(dataset=oaisys_valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=True,
      progress=True,
      num_classes=num_classes,
      aux_loss=aux_loss)
  model.to(device)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])

  if weighted_loss and training_set is not None:
    training_directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{training_set}', 'deeplab')
    num_labels = np.load(os.path.join(training_directory, 'num_labels.npy'))
    weights = torch.zeros(num_classes)
    for i in range(0, num_classes):
      if num_labels[i] != 0:
        weights[i] = 1 / num_labels[i]
    weights *= np.min(num_labels)

  if weighted_loss:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights).to(device)
  else:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_scheduler = LRScheduler(mode='poly',
                             base_lr=lr,
                             nepochs=epochs,
                             iters_per_epoch=len(train_loader),
                             power=.9)
  metric = SegmentationMetric(num_classes)
  wandb.config.update(
    {
      "learning_rate": lr,
      "epochs": epochs,
      "batch_size": batchsize,
    }
  )
  gmm_model = GaussianMixture(n_components=num_classes, n_features=256, covariance_type='diag')
  

  def validation(epoch, best_pred):
    is_best = False
    metric.reset()
    model.eval()
    hooks = get_deeplab_hooks(model, feature_name)
    print("number of valdata: ", len(val_loader))
    for i, (image, target) in enumerate(val_loader):
      image = image.to(device)
      outputs = model(image)['out']
      pred = torch.argmax(outputs, 1)
      pred = pred.cpu().data.numpy()
      features = hooks['feat']
      print('image shape: ', image.shape)
      print('features shape: ', features.shape)
      feature_shape = features.shape
      features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
      print('features shape transposed: ', features.shape)
      assert features.shape[-1] == 256
      features = torch.tensor(features.reshape((-1, 256)))
      print('features shape reshaped: ', features.shape)
      gmm_pred = gmm_model.predict(features).detach().to('cpu').numpy()
      print('gmm_pred shape: ', gmm_pred.shape)
      gmm_pred = gmm_pred.reshape(feature_shape[0], feature_shape[2], feature_shape[3])
      print('gmm_pred shape reshaped: ', gmm_pred.shape)
      gmm_pred_resized = torch.zeros_like(target)
      for i, pred in enumerate(gmm_pred):
        print(i)
        gmm_pred_resized[0,:,:] = torch.tensor(cv2.resize(pred, (images.shape[2], images.shape[3]),
                                interpolation=cv2.INTER_NEAREST))
      print('gmm_pred_resized shape: ', gmm_pred_resized.shape)
      
      metric.update(pred, target.numpy())
    pixAcc, mIoU, IoU = metric.get_result()
    print('Epoch %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' %
          (epoch, pixAcc * 100, mIoU * 100))
    _run.log_scalar('val_miou', mIoU, epoch)
    _run.log_scalar('val_acc', pixAcc, epoch)
    wandb.log({
      'val_miou': mIoU,
      'val_acc': pixAcc,
      'miou_dirt': IoU[0],
      'miou_grass': IoU[1],
      'miou_tree': IoU[2],
      'miou_object': IoU[3],
      'miou_water': IoU[4],
      'miou_sky': IoU[5],
      'miou_gravel': IoU[6],
      'miou_mulch': IoU[7],
      'miou_bed_rock': IoU[8],
      'miou_log': IoU[9],
      'miou_rock': IoU[10],
      'epoch': epoch,
    })

    new_pred = (pixAcc + mIoU) / 2
    if new_pred > best_pred:
      is_best = True
      best_pred = new_pred
      save_checkpoint(model, postfix='best' if is_best else None)

    if separate_eval:
      metric.reset()
      for i, (image, target) in enumerate(rugd_val_loader):
        image = image.to(device)
        outputs = model(image)['out']
        pred = torch.argmax(outputs, 1)
        pred = pred.cpu().data.numpy()
        metric.update(pred, target.numpy())
      pixAcc, mIoU, IoU = metric.get_result()
      wandb.log({
        'rugd_val_miou': mIoU,
        'rugd_val_acc': pixAcc,
        'rugd_miou_dirt': IoU[0],
        'rugd_miou_grass': IoU[1],
        'rugd_miou_tree': IoU[2],
        'rugd_miou_object': IoU[3],
        'rugd_miou_water': IoU[4],
        'rugd_miou_sky': IoU[5],
        'rugd_miou_gravel': IoU[6],
        'rugd_miou_mulch': IoU[7],
        'rugd_miou_bed_rock': IoU[8],
        'rugd_miou_log': IoU[9],
        'rugd_miou_rock': IoU[10],
        'epoch': epoch,
      })
      metric.reset()
      for i, (image, target) in enumerate(oaisys_val_loader):
        image = image.to(device)
        outputs = model(image)['out']
        pred = torch.argmax(outputs, 1)
        pred = pred.cpu().data.numpy()
        metric.update(pred, target.numpy())
      pixAcc, mIoU, IoU = metric.get_result()
      wandb.log({
        'oaisys_val_miou': mIoU,
        'oaisys_val_acc': pixAcc,
        'oaisys_miou_dirt': IoU[0],
        'oaisys_miou_grass': IoU[1],
        'oaisys_miou_tree': IoU[2],
        'oaisys_miou_object': IoU[3],
        'oaisys_miou_water': IoU[4],
        'oaisys_miou_sky': IoU[5],
        'oaisys_miou_gravel': IoU[6],
        'oaisys_miou_mulch': IoU[7],
        'oaisys_miou_bed_rock': IoU[8],
        'oaisys_miou_log': IoU[9],
        'oaisys_miou_rock': IoU[10],
        'epoch': epoch,
      })


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

      hooks = get_deeplab_hooks(model, feature_name)
      outputs = model(images)['out']
      # print(hooks)
      features = hooks['feat']
      feature_shape = features.shape
      features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
      assert features.shape[-1] == 256
      features = torch.tensor(features.reshape((-1, 256)))
      gmm_model.fit(x=features, n_iter=10, warm_start=True, learning_rate=0.01)
      gmm_pred = gmm_model.predict(features).detach().to('cpu').numpy()
      gmm_pred = gmm_pred.reshape(feature_shape[0], feature_shape[2], feature_shape[3])
      gmm_pred_resized = torch.zeros_like(targets)
      for i, pred in enumerate(gmm_pred):
        gmm_pred_resized[0,:,:] = torch.tensor(cv2.resize(pred, (images.shape[2], images.shape[3]),
                                interpolation=cv2.INTER_NEAREST))

      model.train()
      loss = criterion(gmm_pred_resized, targets)

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
      wandb.log({'batch': cur_iters, 'epoch': epoch, 'loss': loss.item(), 'lr': current_lr})
    with torch.no_grad():
      validation(epoch, best_pred)
    if epoch % 2 == 0:
      save_checkpoint(model, postfix=f'{epoch:05d}epochs')
      _run.add_artifact(os.path.join(TMPDIR, f'deeplab_oaisys_1000_test_{epoch:05d}epochs.pth'))

  save_checkpoint(model)

  # upload checkpoints
  for filename in ('deeplab_oaisys_1000_test.pth', 'deeplab_oaisys_1000_test_best.pth'):
    modelpath = os.path.join(TMPDIR, filename)
    _run.add_artifact(modelpath)
  time.sleep(5)
  return best_pred



@ex.command
def deeplab_gmm_oaisys(_run, batchsize, epochs, lr, dataset, aux_loss, device, separate_eval, num_classes, training_set, weighted_loss, use_euler):
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{dataset}.tar')
    if separate_eval:
      os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/rugd.tar')
      os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/oaisys11k.tar')


  wandb.init(project=f'{dataset}-gmm-pretraining')

  # DATA LOADING
  if use_euler:
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
    valdata = tfds.load(dataset,
                        split='validation',
                        as_supervised=True,
                        data_dir=f'{TMPDIR}/datasets')
  else:
    traindata = tfds.load(dataset,
                          split='train',
                          as_supervised=True)
    valdata = tfds.load(dataset,
                        split='validation',
                        as_supervised=True)

  traindata = TFDataIterableDataset(
      traindata.map(lambda x, y: augmentation(
          x, y, random_crop=(256, 256))).map(data_converter_rugd))
  valdata = TFDataIterableDataset(valdata.map(data_converter_rugd))
  train_loader = torch.utils.data.DataLoader(dataset=traindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)
  val_loader = torch.utils.data.DataLoader(dataset=valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)
  if separate_eval:
    if use_euler:
      rugd_valdata = tfds.load('rugd',
                          split='validation',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
      oaisys_valdata = tfds.load('oaisys11k',
                          split='validation',
                          as_supervised=True,
                          data_dir=f'{TMPDIR}/datasets')
    else:
      rugd_valdata = tfds.load('rugd',
                          split='validation',
                          as_supervised=True)
      oaisys_valdata = tfds.load('oaisys11k',
                          split='validation',
                          as_supervised=True)
    rugd_valdata = TFDataIterableDataset(rugd_valdata.map(data_converter_rugd))
    oaisys_valdata = TFDataIterableDataset(oaisys_valdata.map(data_converter_rugd))
    rugd_val_loader = torch.utils.data.DataLoader(dataset=rugd_valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)
    oaisys_val_loader = torch.utils.data.DataLoader(dataset=oaisys_valdata,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=True,
      progress=True,
      num_classes=num_classes,
      aux_loss=aux_loss)
  model.to(device)
  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(
        model, device_ids=[*range(torch.cuda.device_count())])

  if weighted_loss and training_set is not None:
    training_directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{training_set}', 'deeplab')
    num_labels = np.load(os.path.join(training_directory, 'num_labels.npy'))
    weights = torch.zeros(num_classes)
    for i in range(0, num_classes):
      if num_labels[i] != 0:
        weights[i] = 1 / num_labels[i]
    weights *= np.min(num_labels)

  if weighted_loss:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights).to(device)
  else:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  lr_scheduler = LRScheduler(mode='poly',
                             base_lr=lr,
                             nepochs=epochs,
                             iters_per_epoch=len(train_loader),
                             power=.9)
  metric = SegmentationMetric(num_classes)
  wandb.config.update(
    {
      "learning_rate": lr,
      "epochs": epochs,
      "batch_size": batchsize,
    }
  )

  gmm_model = GaussianMixture(n_components=num_classes, n_features=256, covariance_type='diag')

  def validation(epoch, best_pred):
    is_best = False
    metric.reset()
    model.eval()
    print("number of valdata: ", len(val_loader))
    for i, (image, target) in enumerate(val_loader):
      image = image.to(device)
      outputs = model(image)['out']
      pred = torch.argmax(outputs, 1)
      pred = pred.cpu().data.numpy()
      metric.update(pred, target.numpy())
    pixAcc, mIoU, IoU = metric.get_result()
    print('Epoch %d, validation pixAcc: %.3f%%, mIoU: %.3f%%' %
          (epoch, pixAcc * 100, mIoU * 100))
    _run.log_scalar('val_miou', mIoU, epoch)
    _run.log_scalar('val_acc', pixAcc, epoch)
    wandb.log({
      'val_miou': mIoU,
      'val_acc': pixAcc,
      'miou_dirt': IoU[0],
      'miou_grass': IoU[1],
      'miou_tree': IoU[2],
      'miou_object': IoU[3],
      'miou_water': IoU[4],
      'miou_sky': IoU[5],
      'miou_gravel': IoU[6],
      'miou_mulch': IoU[7],
      'miou_bed_rock': IoU[8],
      'miou_log': IoU[9],
      'miou_rock': IoU[10],
      'epoch': epoch,
    })

    new_pred = (pixAcc + mIoU) / 2
    if new_pred > best_pred:
      is_best = True
      best_pred = new_pred
    save_checkpoint(model, postfix='best' if is_best else None)

    if separate_eval:
      metric.reset()
      for i, (image, target) in enumerate(rugd_val_loader):
        image = image.to(device)
        outputs = model(image)['out']
        pred = torch.argmax(outputs, 1)
        pred = pred.cpu().data.numpy()
        metric.update(pred, target.numpy())
      pixAcc, mIoU, IoU = metric.get_result()
      wandb.log({
        'rugd_val_miou': mIoU,
        'rugd_val_acc': pixAcc,
        'rugd_miou_dirt': IoU[0],
        'rugd_miou_grass': IoU[1],
        'rugd_miou_tree': IoU[2],
        'rugd_miou_object': IoU[3],
        'rugd_miou_water': IoU[4],
        'rugd_miou_sky': IoU[5],
        'rugd_miou_gravel': IoU[6],
        'rugd_miou_mulch': IoU[7],
        'rugd_miou_bed_rock': IoU[8],
        'rugd_miou_log': IoU[9],
        'rugd_miou_rock': IoU[10],
        'epoch': epoch,
      })
      metric.reset()
      for i, (image, target) in enumerate(oaisys_val_loader):
        image = image.to(device)
        outputs = model(image)['out']
        pred = torch.argmax(outputs, 1)
        pred = pred.cpu().data.numpy()
        metric.update(pred, target.numpy())
      pixAcc, mIoU, IoU = metric.get_result()
      wandb.log({
        'oaisys_val_miou': mIoU,
        'oaisys_val_acc': pixAcc,
        'oaisys_miou_dirt': IoU[0],
        'oaisys_miou_grass': IoU[1],
        'oaisys_miou_tree': IoU[2],
        'oaisys_miou_object': IoU[3],
        'oaisys_miou_water': IoU[4],
        'oaisys_miou_sky': IoU[5],
        'oaisys_miou_gravel': IoU[6],
        'oaisys_miou_mulch': IoU[7],
        'oaisys_miou_bed_rock': IoU[8],
        'oaisys_miou_log': IoU[9],
        'oaisys_miou_rock': IoU[10],
        'epoch': epoch,
      })


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
      print(outputs.size)

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
      wandb.log({'batch': cur_iters, 'epoch': epoch, 'loss': loss.item(), 'lr': current_lr})
    with torch.no_grad():
      validation(epoch, best_pred)
    if epoch % 2 == 0:
      save_checkpoint(model, postfix=f'{epoch:05d}epochs')
      _run.add_artifact(os.path.join(TMPDIR, f'deeplab_oaisys_1000_test_{epoch:05d}epochs.pth'))

  save_checkpoint(model)

  # upload checkpoints
  for filename in ('deeplab_oaisys_1000_test.pth', 'deeplab_oaisys_1000_test_best.pth'):
    modelpath = os.path.join(TMPDIR, filename)
    _run.add_artifact(modelpath)
  time.sleep(5)
  return best_pred


if __name__ == '__main__':
  ex.run_commandline()
