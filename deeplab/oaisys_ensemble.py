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

import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)

from semsegcluster.data.images import convert_img_to_float
from semsegcluster.settings import EXP_OUT, TMPDIR
from semsegcluster.sacred_utils import get_checkpoint
from semsegcluster.data.tfds_to_torch import TFDataIterableDataset
from deeplab.oaisys_utils import data_converter_rugd, OAISYS_LABELS, load_checkpoint

ex = Experiment()

@ex.main
def run_deeplab_ensemble(
    pretrained_models=["/cluster/scratch/loewsi/scimfolder/logs/282/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/283/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/284/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/285/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/286/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/287/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/288/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/289/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/290/deeplab_oaisys_1000_test_00004epochs.pth","/cluster/scratch/loewsi/scimfolder/logs/291/deeplab_oaisys_1000_test_00004epochs.pth"],
    split='validation',
    device='cuda', 
    dataname='oaisys_trajectory', 
    postfix='ensemble',
    num_classes=11,
    use_euler=False,
):
  # DATA SETUP
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{dataname}.tar')
    data = tfds.load(f'{dataname}', split=split, data_dir=f'{TMPDIR}/datasets', as_supervised=True)
  else:
    data = tfds.load(f'{dataname}', split=split, as_supervised=True)
    pretrained_models=['/home/asl/Downloads/file_transfer/logs/282/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/283/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/284/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/285/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/286/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/287/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/288/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/289/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/290/deeplab_oaisys_1000_test_00004epochs.pth', '/home/asl/Downloads/file_transfer/logs/291/deeplab_oaisys_1000_test_00004epochs.pth']
  dataset = TFDataIterableDataset(data.map(data_converter_rugd))
  data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

  # MODEL SETUP
  models = []
  for i in range(len(pretrained_models)):
    model = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False,
        pretrained_backbone=False,
        progress=True,
        num_classes=num_classes,
        aux_loss=None)
    print(pretrained_models[i])
    checkpoint, pretrained_id = get_checkpoint(pretrained_models[i])

    # remove any aux classifier stuff
    removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
    for k in removekeys:
      del checkpoint[k]
    load_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()
    models.append(model)
  # DIRECTORY SETUP
  directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{dataname}', pretrained_id)
  os.makedirs(directory, exist_ok=True)

  cm = torchmetrics.ConfusionMatrix(num_classes=20, compute_on_step=False)

  # INFERENCE
  for idx, (image, label) in enumerate(tqdm(data_loader)):
    # run inference
    for i in range(len(pretrained_models)):
      model = models[i]
      logits = model(image.to(device))['out']
      if i == 0:
        logits_sum = logits
      else:
        logits_sum += logits
    logits = logits_sum / len(pretrained_models)
    max_logit, pred = torch.max(logits, 1)
    
    # update confusion matrix
    valid_pred = pred.detach().to('cpu')[label != 255]
    valid_label = label[label != 255]
    cm.update(valid_pred, valid_label)

    # store outputs
    name = f'{idx}_{postfix}'
    np.save(os.path.join(directory, f'{name}_pred.npy'),
            pred[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{name}_maxlogit-pp.npy'),
            -max_logit[0].detach().to('cpu').numpy())
    np.save(os.path.join(directory, f'{idx}_label.npy'), label)

  # CONFUSION MATRIX
  cm = cm.compute().numpy()
  np.save(os.path.join(directory, f'confusion_matrix_{postfix}.npy'), cm)
  disp = ConfusionMatrixDisplay(cm / cm.sum(0),
                                display_labels=OAISYS_LABELS)
  plt.figure(figsize=(20, 20))
  disp.plot(ax=plt.gca(), xticks_rotation='vertical', include_values=False)
  plt.savefig(os.path.join(directory, f'confusion_matrix_{postfix}.pdf'))


if __name__ == '__main__':
  ex.run_commandline()
