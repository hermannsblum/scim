from sacred import Experiment
import torch
import torchvision
import torchmetrics
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

tf.config.set_visible_devices([], 'GPU')

from semsegcluster.data.images import convert_img_to_float
from semsegcluster.settings import EXP_OUT
from semsegcluster.sacred_utils import get_checkpoint
from semsegcluster.model.postprocessing import BoundarySuppressionWithSmoothing, standardize_max_logits
from semsegcluster.data.tfds_to_torch import TFDataIterableDataset


ex = Experiment()

OAISYS_LABELS = [
    'Dirt', 'Grass', 'Tree', 'Object', 'Water', 'Sky', 'Gravel', 'Mulch', 'Bedrock', 'Log', 'Rock', 'Empty_1', 'Empty_2', 'Empty_4', 'Empty_5', 'Sand'
]

def data_converter_rugd(image, label):
  image = convert_img_to_float(image)
  label = tf.squeeze(tf.cast(label, tf.int64))

  label = tf.where(label == 0, tf.cast(255, tf.int64), label)
  label = tf.where(label == 1, tf.cast(0, tf.int64), label)
  label = tf.where(label == 3, tf.cast(1, tf.int64), label)
  label = tf.where(label == 9, tf.cast(3, tf.int64), label)
  label = tf.where(label == 15, tf.cast(9, tf.int64), label)
  label = tf.where(label == 2, tf.cast(15, tf.int64), label)
  label = tf.where(label == 4, tf.cast(2, tf.int64), label)
  label = tf.where(label == 5, tf.cast(3, tf.int64), label)
  label = tf.where(label == 6, tf.cast(4, tf.int64), label)
  label = tf.where(label == 7, tf.cast(5, tf.int64), label)
  label = tf.where(label == 8, tf.cast(3, tf.int64), label)
  label = tf.where(label == 10, tf.cast(255, tf.int64), label)
  label = tf.where(label == 11, tf.cast(6, tf.int64), label)
  label = tf.where(label == 12, tf.cast(255, tf.int64), label)
  label = tf.where(label == 13, tf.cast(7, tf.int64), label)
  label = tf.where(label == 14, tf.cast(8, tf.int64), label)
  label = tf.where(label == 16, tf.cast(3, tf.int64), label)
  label = tf.where(label == 17, tf.cast(3, tf.int64), label)
  label = tf.where(label == 18, tf.cast(3, tf.int64), label)
  label = tf.where(label == 19, tf.cast(2, tf.int64), label)
  label = tf.where(label == 20, tf.cast(3, tf.int64), label)
  label = tf.where(label == 21, tf.cast(10, tf.int64), label)
  label = tf.where(label == 22, tf.cast(255, tf.int64), label)
  label = tf.where(label == 23, tf.cast(255, tf.int64), label)
  label = tf.where(label == 24, tf.cast(3, tf.int64), label)
  label = tf.where(label > 15, tf.cast(1, tf.int64), label)  # grass label where labeled wrong by oaisys postprocessing

  # move channel from last to 2nd
  image = tf.transpose(image, perm=[2, 0, 1])
  return image, label


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


@ex.command
def run_sml(fitting_experiment, subset, device='cuda'):
  raise NotImplementedError("run_sml not implemented for OAISYS dataset yet!")
  


@ex.command
def run_knn(
    fitting_experiment,
    subset,
    device='cuda',
    ef=200,
    k=1,
    feature_name='classifier.2',
):
  raise NotImplementedError("run_knn not implemented for OAISYS dataset yet!")


@ex.main
def run_deeplab(
    pretrained_model, 
    split='validation', 
    device='cuda', 
    set='oaisys_trajectory', 
    ignore_other=True, 
    normalize_max_logits=True, 
    training_set=None, 
    validation_plots=False
):
  # WRITER SETUP
  if validation_plots:
    writer = SummaryWriter()
  
  # DATA SETUP
  data = tfds.load(f'{set}', split=split, as_supervised=True)
  dataset = TFDataIterableDataset(data.map(data_converter_rugd))
  data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=15,
      aux_loss=None)
  checkpoint, pretrained_id = get_checkpoint(pretrained_model)
  # remove any aux classifier stuff
  removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
  for k in removekeys:
    del checkpoint[k]
  load_checkpoint(model, checkpoint)
  model.to(device)
  model.eval()
  postprocessing = BoundarySuppressionWithSmoothing().to(device)

  # DIRECTORY SETUP
  directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{set}', pretrained_id)
  os.makedirs(directory, exist_ok=True)

  cm = torchmetrics.ConfusionMatrix(num_classes=16, compute_on_step=False)

  # INFERENCE
  for idx, (image, label) in enumerate(tqdm(data_loader)):
    # run inference
    logits = model(image)['out']
    max_logit, pred = torch.max(logits, 1)
    softmax_entropy = torch.distributions.categorical.Categorical(
        logits=logits.permute(0, 2, 3, 1)).entropy()
    # postprocessing
    if normalize_max_logits and training_set is not None:
      # load mean and sigma of logits
      training_directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{training_set}', pretrained_id)
      mean_logits = np.load(os.path.join(training_directory, 'mean_max_logit.npy'))
      sigma_logits = np.load(os.path.join(training_directory, 'sigma_max_logit.npy'))
      max_logit = standardize_max_logits(pred=pred, logits=max_logit, mean=mean_logits, sigma=sigma_logits)
    elif normalize_max_logits and training_set is None:
      raise ValueError("Cannot normalize max logits without training set!")
    max_logit = postprocessing(max_logit, prediction=pred)
    
    # update confusion matrix
    valid_pred = pred.detach().to('cpu')[label != 255]
    valid_label = label[label != 255]
    cm.update(valid_pred, valid_label)

    # store outputs
    name = idx
    np.save(os.path.join(directory, f'{name}_pred.npy'),
            pred[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_entropy.npy'),
            softmax_entropy[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_maxlogit-pp.npy'),
            -max_logit[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_label.npy'), label)

    if validation_plots:
      writer.add_image('image_loaded', image.numpy()[0], idx, dataformats='CHW')
      writer.add_image('labels_loaded', label.numpy()[0] , idx, dataformats='HW')
      writer.add_image('prediction', pred[0].detach().to('cpu').numpy(), idx, dataformats="HW")
      writer.add_image('softmax entropy', softmax_entropy[0].detach().to('cpu').numpy(), idx, dataformats="HW")
      writer.add_image('- max logit', -max_logit[0].detach().to('cpu').numpy(), idx, dataformats="HW")
      writer.close()

  # CONFUSION MATRIX
  cm = cm.compute().numpy()
  np.save(os.path.join(directory, 'confusion_matrix.npy'), cm)
  disp = ConfusionMatrixDisplay(cm / cm.sum(0),
                                display_labels=OAISYS_LABELS)
  plt.figure(figsize=(20, 20))
  disp.plot(ax=plt.gca(), xticks_rotation='vertical', include_values=False)
  plt.savefig(os.path.join(directory, 'confusion_matrix.pdf'))


if __name__ == '__main__':
  ex.run_commandline()
