from sacred import Experiment
import torch
import tensorflow_datasets as tfds
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from semseg_density.data.images import convert_img_to_float
from semseg_density.model.refinenet import rf_lw50, rf_lw101
from semseg_density.model.refinenet_uncertainty import RefineNetDensity
from semseg_density.settings import TMPDIR

ex = Experiment()


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


@ex.main
def run_scannet_inference(pretrained_model,
                          subset,
                          n_components,
                          feature_layer='mflow_conv_g4_pool',
                          size=50,
                          device='cuda',
                          groupnorm=False):
  data = tfds.load(f'scan_net/{subset}', split='train')

  # MODEL SETUP
  if not size in [50, 101]:
    raise UserWarning("Unknown model size.")
  model = RefineNetDensity(40,
                           size=size,
                           n_components=n_components,
                           feature_layer=feature_layer)
  # Load pretrained weights
  if pretrained_model and pretrained_model != 'adelaine':
    checkpoint = torch.load(load_gdrive_file(pretrained_model, ending='pth'))
    load_checkpoint(model, checkpoint, strict=False)
  model.to(device)
  model.eval()

  for blob in data:
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])

    label = tf.cast(blob['label'], tf.int64)
    # the output is 4 times smaller than the input, so transform labels
    label = tf.image.resize(label[..., tf.newaxis], (120, 160),
                            method='nearest')[..., 0]

    name = blob['name'].numpy().decode()
    print(name)

    break
