from sacred import Experiment
import cv2
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
from deeplab.marsscapes_utils import data_converter_marsscapes

ex = Experiment()


@ex.main
def run_deeplab(
    pretrained_model, 
    device='cuda', 
    validation_plots=False,
    postfix='',
    num_classes=15,
):
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
  directory = os.path.join(EXP_OUT, 'marsscapes_inference', pretrained_id)
  os.makedirs(directory, exist_ok=True)

  cm = torchmetrics.ConfusionMatrix(num_classes=16, compute_on_step=False)

  # INFERENCE
  test_path = '/home/asl/Downloads/MarsScapes/processed/test/'
  val_path = '/home/asl/Downloads/MarsScapes/processed/val/'
  train_path = '/home/asl/Downloads/MarsScapes/processed/train/'
  for image_name in tqdm(os.listdir(test_path)):
    # load image and label
    if not (image_name.endswith(".png")):
      continue
    if 'color' in image_name:
      continue
    if 'semanticId' in image_name:
      continue
    image = cv2.cvtColor(cv2.imread(os.path.join(test_path, image_name)), cv2.COLOR_RGB2BGR)/255.
    image = torch.tensor(image).permute([2, 0, 1])[None,:].type(torch.FloatTensor)

    label_name = image_name[:-4] + '_semanticId.png'
    if os.path.exists(os.path.join(test_path, label_name)):
      label = cv2.cvtColor(cv2.imread(os.path.join(test_path, label_name)), cv2.COLOR_RGB2BGR)[:,:,0]
    elif os.path.exists(os.path.join(val_path, label_name)):
      label = cv2.cvtColor(cv2.imread(os.path.join(val_path, label_name)), cv2.COLOR_RGB2BGR)[:,:,0]
    elif os.path.exists(os.path.join(train_path, label_name)):
      label = cv2.cvtColor(cv2.imread(os.path.join(train_path, label_name)), cv2.COLOR_RGB2BGR)[:,:,0]
    else:
      print(f'Label {label_name} not found')
      continue
    label = torch.tensor(label)[None,:]
    label = data_converter_marsscapes(label).type(torch.LongTensor)
    # run inference
    logits = model(image)['out']
    max_logit, pred = torch.max(logits, 1)
    softmax_entropy = torch.distributions.categorical.Categorical(
        logits=logits.permute(0, 2, 3, 1)).entropy()
    # postprocessing
    max_logit = postprocessing(max_logit, prediction=pred)
    
    # update confusion matrix
    valid_pred = pred.detach().to('cpu')[label != 255]
    valid_label = label[label != 255]
    cm.update(valid_pred, valid_label)

    # store outputs
    return
    name = image_name[:-4]
    np.save(os.path.join(directory, f'{name}_pred.npy'),
            pred[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_entropy.npy'),
            softmax_entropy[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_maxlogit-pp.npy'),
            -max_logit[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_label.npy'), label)


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
