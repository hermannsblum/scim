from sacred import Experiment
import torch
import torchvision
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import cv2
from datetime import datetime

tf.config.set_visible_devices([], 'GPU')
import os
import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)
import time
from collections import OrderedDict
from shutil import copyfile, copytree

from semsegcluster.data.tfds_to_torch import TFDataIterableDataset
from semsegcluster.data.augmentation import augmentation
from semsegcluster.data.images import convert_img_to_float
from semsegcluster.lr_scheduler import LRScheduler
from semsegcluster.segmentation_metrics import SegmentationMetric
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_observer, get_checkpoint
from deeplab.oaisys_utils import data_converter_rugd

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
  filename = 'deeplab_oaisys_adapted.pth'
  save_path = os.path.join(TMPDIR, filename)
  if hasattr(model, 'module'):
    torch.save(model.module.state_dict(), save_path)
  else:
    torch.save(model.state_dict(), save_path)
  if postfix is not None:
    best_filename = f'deeplab_oaisys_adapted_{postfix}.pth'
    best_filename = os.path.join(TMPDIR, best_filename)
    copyfile(save_path, best_filename)


ex.add_config(
    pretrained_models=["/cluster/scratch/loewsi/scimfolder/logs/282/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/283/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/284/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/285/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/286/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/287/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/288/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/289/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/290/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/291/deeplab_oaisys_1000_test_00004epochs.pth"],
    batchsize=10,
    epochs=30,
    lr=1e-5,
    keep_threshold=1e4,
    uhlemeyer_only_single_class=True,
    aux_loss=None,
    use_euler=False,
    num_classes=11,
    pretrained_dataset='oaisys16k_rugd',
)


@ex.main
def train(
    _run,
    batchsize,
    epochs,
    pretrained_models,
    lr,
    aux_loss,
    use_euler,
    subset,
    pretrain_dataset,
    pseudolabels,
    keep_threshold,
    uhlemeyer_only_single_class,
    num_classes,
    device='cuda',
    num_valdata=60,
):
  # DATA LOADING
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{subset}.tar')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{pretrain_dataset}.tar')
    data = tfds.load(f'{subset}', split='validation', data_dir=f'{TMPDIR}/datasets')
    pretraindata = tfds.load(pretrain_dataset,
                        split='train',
                        as_supervised=True,
                        data_dir=f'{TMPDIR}/datasets').shuffle(len(data)-60)
  else:
    data = tfds.load(f'{subset}', split='validation')
    pretraindata = tfds.load(pretrain_dataset,
                        split='train',
                        as_supervised=True).shuffle(len(data)-60)
  _, pretrained_id = get_checkpoint(pretrained_models[0])
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  # check if labels are PNG or NPY
  is_png = os.path.exists(
      os.path.join(directory, f'{0:06d}_{pseudolabels}.png'))

  # filter labels > num_classes for those that actually exist
  label_counts = np.zeros(1, dtype=np.int64)
  for idx in range(len(data)):
    if is_png:
      label = cv2.imread(os.path.join(directory, f'{idx:06d}_{pseudolabels}.png'),
                         cv2.IMREAD_ANYDEPTH).squeeze().astype(np.int32)
      # label -= 1
    else:
      label = np.load(os.path.join(directory,
                                   f'{idx:06d}_{pseudolabels}.npy')).squeeze()
    count = np.bincount(label[label >= num_classes])
    if count.shape[0] > label_counts.shape[0]:
      label_counts = np.pad(label_counts,
                            ((0, count.shape[0] - label_counts.shape[0])))
    else:
      count = np.pad(count, ((0, label_counts.shape[0] - count.shape[0])))
    label_counts += count
  label_map = np.arange(label_counts.shape[0] + 1)
  min_count = max(keep_threshold,
                  np.partition(label_counts.flatten(), -10)[-10])
  new_cluster = num_classes
  for i in range(num_classes, label_counts.shape[0]):
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
    for idx, blob in enumerate(data):
      image = convert_img_to_float(blob['image'])
      if is_png:
        label = cv2.imread(
            os.path.join(directory, f'{idx:06d}_{pseudolabels}.png'),
            cv2.IMREAD_ANYDEPTH).squeeze().astype(np.int32)
        # label -= 1
      else:
        label = np.load(os.path.join(directory,
                                     f'{idx:06d}_{pseudolabels}.npy')).squeeze()
      label = label_map[label]
      yield image, label

  gendata = tf.data.Dataset.from_generator(
      data_generator,
      output_signature=(tf.TensorSpec(shape=(480, 640, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(480, 640),
                                      dtype=tf.int32))).shuffle(1000)
  pretraindata = TFDataIterableDataset(
      pretraindata.map(lambda x, y: augmentation(
          x, y, random_crop=(256, 256))).map(data_converter_rugd))

  valdata = gendata.take(num_valdata)
  traindata = gendata.skip(num_valdata)

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
  pretrain_loader = torch.utils.data.DataLoader(dataset=pretraindata,
                                             batch_size=batchsize,
                                             pin_memory=True,
                                             drop_last=True)

  for model_idx, pretrained_model in enumerate(pretrained_models):
    # MODEL SETUP
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        pretrained_backbone=False,
        progress=True,
        num_classes=21,
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
    metric = SegmentationMetric(25)

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
      save_checkpoint(model, postfix=f'best_model_{model_idx}' if is_best else None)

    best_pred = .0
    cur_iters = 0
    start_time = time.time()
    for epoch in range(epochs):
      model.train()

      for i, ((images, targets), (pretrain_images, pretrain_targets)) in enumerate(zip(train_loader, pretrain_loader)):
        # learning-rate update
        current_lr = lr_scheduler(cur_iters)
        for param_group in optimizer.param_groups:
          param_group['lr'] = current_lr

        # train new images
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)['out']
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train pretrain images
        images = pretrain_images.to(device)
        targets = pretrain_targets.to(device)

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
      if epoch % 2 == 0:
        save_checkpoint(model, postfix=f'{epoch}epochs_model_{model_idx}')

    save_checkpoint(model, postfix=f'final_model_{model_idx}')

    # upload checkpoint
    _run.add_artifact(os.path.join(TMPDIR, f'deeplab_oaisys_adapted_best_model_{model_idx}.pth'))
  time.sleep(5)
  if use_euler:
    copytree(TMPDIR, f'{directory}/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    time.sleep(5)
  return best_pred


if __name__ == '__main__':
  ex.run_commandline()
