from __future__ import division

import threading
import numpy as np

__all__ = [
    'SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
    'hist_info', 'compute_score'
]
"""Evaluation Metrics for Semantic Segmentation"""


class SegmentationMetric(object):
  """Computes pixAcc and mIoU metric scores
    """

  def __init__(self, nclass):
    super(SegmentationMetric, self).__init__()
    self.nclass = nclass
    self.lock = threading.Lock()
    self.reset()

  def update(self, preds, labels):
    """Updates the internal evaluation result.
        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """
    if isinstance(preds, np.ndarray):
      self.evaluate_worker(preds, labels)
    elif isinstance(preds, (list, tuple)):
      threads = [
          threading.Thread(
              target=self.evaluate_worker,
              args=(pred, label),
          ) for (pred, label) in zip(preds, labels)
      ]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

  def get(self):
    """Gets the current evaluation result.
        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
    pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
    IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
    # It has same result with np.nanmean() when all class exist
    mIoU = IoU.mean()
    return pixAcc, mIoU

  def evaluate_worker(self, pred, label):
    correct, labeled = batch_pix_accuracy(pred, label)
    inter, union = batch_intersection_union(pred, label, self.nclass)
    with self.lock:
      self.total_correct += correct
      self.total_label += labeled
      self.total_inter += inter
      self.total_union += union

  def reset(self):
    """Resets the internal evaluation result to initial state."""
    self.total_inter = 0
    self.total_union = 0
    self.total_correct = 0
    self.total_label = 0


def batch_pix_accuracy(predict, target):
  """PixAcc"""
  # inputs are numpy array, output 4D, target 3D
  assert predict.shape == target.shape
  predict = predict.astype('int64') + 1
  target = target.astype('int64') + 1

  pixel_labeled = np.logical_and(target > 0, target != 256)
  n_pixel_labeled = np.sum(pixel_labeled)
  n_pixel_correct = np.sum((predict == target) * pixel_labeled)
  assert n_pixel_correct <= n_pixel_labeled, "Correct area should be smaller than Labeled"
  return n_pixel_correct, n_pixel_labeled


def batch_intersection_union(predict, target, nclass):
  """mIoU"""
  # inputs are numpy array, output 4D, target 3D
  assert predict.shape == target.shape
  mini = 1
  maxi = nclass
  nbins = nclass
  predict = predict.astype('int64') + 1
  target = target.astype('int64') + 1

  pixel_labeled = np.logical_and(target > 0, target != 256)
  predict = predict * pixel_labeled.astype(predict.dtype)
  intersection = predict * (predict == target)
  # areas of intersection and union
  # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
  area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
  area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
  area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
  area_union = area_pred + area_lab - area_inter
  assert (area_inter <= area_union
         ).all(), "Intersection area should be smaller than Union area"
  return area_inter, area_union


def hist_info(pred, label, num_cls):
  assert pred.shape == label.shape
  k = (label >= 0) & (label < num_cls)
  labeled = np.sum(k)
  correct = np.sum((pred[k] == label[k]))

  return np.bincount(num_cls * label[k].astype(int) + pred[k],
                     minlength=num_cls**2).reshape(num_cls,
                                                   num_cls), labeled, correct


def compute_score(hist, correct, labeled):
  iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
  # print('right')
  # print(iu)
  mean_IU = np.nanmean(iu)
  mean_IU_no_back = np.nanmean(iu[1:])
  freq = hist.sum(1) / hist.sum()
  freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
  mean_pixel_acc = correct / labeled

  return iu, mean_IU, mean_IU_no_back, mean_pixel_acc
