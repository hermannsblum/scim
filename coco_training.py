from sacred import Experiment
import torch
import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import os
import time
from shutil import make_archive, copyfile

import fastscnn.data.coco_segmentation
from fastscnn.data.tfds_to_torch import TFDataIterableDataset
from fastscnn.model import FastSCNN
from fastscnn.lr_scheduler import LRScheduler
from fastscnn.segmentation_metrics import SegmentationMetric
from fastscnn.losses import MixSoftmaxCrossEntropyLoss
from fastscnn.data.images import augmentation
from fastscnn.settings import TMPDIR
from fastscnn.sacred_utils import get_observer

ex = Experiment()
ex.observers.append(get_observer())


def save_checkpoint(model, is_best=False):
    """Save Checkpoint"""
    filename = 'fastscnn_coco.pth'
    save_path = os.path.join(TMPDIR, filename)
    torch.save(model.state_dict(), save_path)
    if is_best:
        best_filename = 'fastscnn_coco_best.pth'
        best_filename = os.path.join(TMPDIR, best_filename)
        copyfile(save_path, best_filename)


@ex.main
def train(_run, batchsize=10, epochs=100, learning_rate=1e-4, device='cuda'):
    # DATA LOADING
    traindata = tfds.load('coco_segmentation',
                          split='train',
                          as_supervised=True)
    valdata = tfds.load('coco_segmentation',
                        split='validation',
                        as_supervised=True)

    def data_converter(image, label):
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int64)
        # move channel from last to 2nd
        image = tf.transpose(image, perm=[2, 0, 1])
        # remove last channel of label
        label = label[..., 0]
        return image, label

    traindata = TFDataIterableDataset(
         traindata.map(data_converter).prefetch(10000))
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
    model = FastSCNN(133)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[*range(torch.cuda.device_count())])
    model.to(device)

    criterion = MixSoftmaxCrossEntropyLoss(ignore_label=255).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = LRScheduler(mode='poly',
                               base_lr=learning_rate,
                               nepochs=epochs,
                               iters_per_epoch=len(train_loader),
                               power=.9)
    metric = SegmentationMetric(133)

    def validation(epoch, best_pred):
        is_best = False
        metric.reset()
        model.eval()
        for i, (image, target) in enumerate(val_loader):
            image = image.to(device)
            outputs = model(image)
            pred = torch.argmax(outputs[0], 1)
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
        save_checkpoint(model, is_best)

    best_pred = .0
    cur_iters = 0
    start_time = time.time()
    for epoch in range(epochs):
        model.train()

        for i, (images, targets) in enumerate(train_loader):
            cur_lr = lr_scheduler(cur_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

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
                    'Epoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.8f || Loss: %.4f'
                    % (epoch, epochs, i + 1, len(train_loader),
                       time.time() - start_time, cur_lr, loss.item()), flush=True)
        _run.log_scalar('loss', loss.item(), epoch)
        _run.log_scalar('learningrate', cur_lr, epoch)
        validation(epoch, best_pred)

    save_checkpoint(model, is_best=False)

    # upload checkpoints
    for filename in ('fastscnn_coco.pth', 'fastscnn_coco_best.pth'):
        modelpath = os.path.join(TMPDIR, filename)
        _run.add_artifact(modelpath)
    return best_pred


if __name__ == '__main__':
    ex.run_commandline()
