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
from deeplab.oaisys_utils import data_converter_rugd, OAISYS_LABELS, load_checkpoint

ex = Experiment()

@ex.main
def run_deeplab(
    pretrained_model, 
    split='validation', 
    device='cuda', 
    set='oaisys_trajectory', 
    ignore_other=True, 
    normalize_max_logits=False, 
    training_set=None, 
    validation_plots=False,
    postfix='',
    num_classes=15,
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
      num_classes=num_classes,
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
      if postfix == '':
        postfix = '_norm'
    elif normalize_max_logits and training_set is None:
      raise ValueError("Cannot normalize max logits without training set!")
    max_logit = postprocessing(max_logit, prediction=pred)
    
    # update confusion matrix
    valid_pred = pred.detach().to('cpu')[label != 255]
    valid_label = label[label != 255]
    cm.update(valid_pred, valid_label)

    # store outputs
    name = f'{idx}{postfix}'
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
  np.save(os.path.join(directory, f'confusion_matrix{postfix}.npy'), cm)
  disp = ConfusionMatrixDisplay(cm / cm.sum(0),
                                display_labels=OAISYS_LABELS)
  plt.figure(figsize=(20, 20))
  disp.plot(ax=plt.gca(), xticks_rotation='vertical', include_values=False)
  plt.savefig(os.path.join(directory, f'confusion_matrix{postfix}.pdf'))


if __name__ == '__main__':
  ex.run_commandline()
