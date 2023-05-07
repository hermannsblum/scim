import torch
import torchvision
import PIL
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
import pickle
from tqdm import tqdm
import os
import cv2
from joblib import Memory
import pickle
from collections import OrderedDict

from semsegcluster.data.images import convert_img_to_float
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_checkpoint
from oaisys_training_euler import data_converter_rugd

memory = Memory(EXP_OUT)


def load_checkpoint(model, state_dict, strict=True):
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


def get_deeplab(pretrained_model, feature_name, device, num_classes=15):
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=num_classes,
      aux_loss=False)
  checkpoint, pretrained_id = get_checkpoint(pretrained_model)
  # remove any aux classifier stuff
  removekeys = [k for k in checkpoint.keys() if k.startswith('aux_classifier')]
  for k in removekeys:
    del checkpoint[k]
  load_checkpoint(model, checkpoint)
  model.to(device)
  model.eval()

  # Create hook to get features from intermediate pytorch layer
  hooks = {}

  def get_activation(name, features=hooks):

    def hook(model, input, output):
      features['feat'] = output.detach()

    return hook

  # register hook to get features
  for n, m in model.named_modules():
    if n == feature_name:
      m.register_forward_hook(get_activation(feature_name))
  return model, hooks

def get_deeplab_hooks(model, feature_name):
  # Create hook to get features from intermediate pytorch layer
  hooks = {}

  def get_activation(name, features=hooks):

    def hook(model, input, output):
      features['feat'] = output.detach()

    return hook

  # register hook to get features
  for n, m in model.named_modules():
    if n == feature_name:
      m.register_forward_hook(get_activation(feature_name))
  return hooks



def get_dino(device):
  vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
  vitb8.to(device)
  vitb8.eval()
  return vitb8


def dino_color_normalize(x,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.228, 0.224, 0.225]):
  for t, m, s in zip(x, mean, std):
    t.sub_(m)
    t.div_(s)
  return x


@memory.cache
def get_sampling_idx(subset, shard, subsample, pretrained_model,
                     expected_feature_shape, use_euler, use_mapping=True, split='validation'):
  rng = default_rng(35643243)
  if use_euler:
    data = tfds.load(f'{subset}', split=split, data_dir=f'{TMPDIR}/datasets')
  else:
    data = tfds.load(f'{subset}', split=split)
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)

  sampling_idx = {}
  all_voxels = np.array([], dtype=np.int32)
  for idx in tqdm(range(len(data))):
    frame = f'{idx}'
    sampled_idx = np.array([], dtype=np.int32)
    # now add random samples
    sampled_idx = np.concatenate(
        (sampled_idx,
         rng.choice(expected_feature_shape[0] * expected_feature_shape[1],
                    size=[subsample - sampled_idx.shape[0]],
                    replace=False)),
        axis=0)
    sampling_idx[frame] = sampled_idx

  return {
      'voxels': all_voxels,
      'sampling_idx': sampling_idx,
  }


@memory.cache
def get_deeplab_embeddings(subset, shard, subsample, device, pretrained_model,
                           expected_feature_shape, feature_name, pred_name,
                           uncert_name, use_euler, use_mapping=True, num_classes=15):
  if use_euler:
    data = tfds.load(f'{subset}', split='validation', data_dir=f'{TMPDIR}/datasets')
  else:
    data = tfds.load(f'{subset}', split='validation')
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  sampling_idx = {}
  sampling_idx = get_sampling_idx(
      subset=subset,
      shard=shard,
      subsample=subsample,
      expected_feature_shape=expected_feature_shape,
      pretrained_model=pretrained_model,
      use_euler=use_euler,
      use_mapping=use_mapping)['sampling_idx']
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
                             device=device,
                             num_classes=num_classes)
  all_features = []
  all_labels = []
  all_uncertainties = []
  all_entropies = []
  for idx, blob in tqdm(enumerate(data)):
    frame = f'{idx}'
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)
    # run inference
    logits = model(image)['out']
    entropy = torch.distributions.categorical.Categorical(
        logits=logits.permute(0, 2, 3, 1)).entropy()
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    assert features.shape[1] == expected_feature_shape[0]
    assert features.shape[2] == expected_feature_shape[1]
    label = np.load(os.path.join(directory,
                                 f'{frame}_{pred_name}.npy')).squeeze()
    uncert = np.load(os.path.join(directory,
                                  f'{frame}_{uncert_name}.npy')).squeeze()    # interpolate to feature size

    label = cv2.resize(label,
                       dsize=(features.shape[2], features.shape[1]),
                       interpolation=cv2.INTER_NEAREST)
    feature_shape = features.shape[1:3]
    assert label.shape[0] == feature_shape[0]
    assert label.shape[1] == feature_shape[1]
    label = label.flatten()
    uncert = cv2.resize(uncert,
                        dsize=(features.shape[2], features.shape[1]),
                        interpolation=cv2.INTER_LINEAR).flatten()
    entropy = torchvision.transforms.functional.resize(
        entropy,
        size=(features.shape[1], features.shape[2]),
        interpolation=PIL.Image.BILINEAR).to('cpu').detach().numpy().flatten()
    features = features.reshape((-1, 256))
    samples = sampling_idx[frame]
    all_features.append(features[samples])
    all_labels.append(label[samples])
    all_uncertainties.append(uncert[samples])
    all_entropies.append(entropy[samples])
    del logits, image, features, label, uncert

  out_features = np.array([[]])
  out_prediction = np.array([[]])
  out_uncertainty = np.array([[]])
  out_entropy = np.array([[]])
  if all_features:
    out_features = np.concatenate(all_features, axis=0)
  else:
    print("No features found, returning empty array")
  if all_labels:
    out_prediction = np.concatenate(all_labels, axis=0)
  else:
    print("No features found, returning empty array")
  if all_uncertainties:
    out_uncertainty = np.concatenate(all_uncertainties, axis=0)
  else:
    print("No features found, returning empty array")
  if all_entropies:
    out_entropy = np.concatenate(all_entropies, axis=0)
  else:
    print("No features found, returning empty array")

  return {
      'features': out_features,
      'prediction': out_prediction,
      'uncertainty': out_uncertainty,
      'entropy': out_entropy,
  }

@memory.cache
def get_deeplab_embeddings_features_labels(subset, shard, subsample, device, pretrained_model,
                           expected_feature_shape, feature_name, use_euler, use_mapping=True, split='validation'):
  if use_euler:
    data = tfds.load(f'{subset}', split=split, data_dir=f'{TMPDIR}/datasets')
  else:
    data = tfds.load(f'{subset}', split=split)
  sampling_idx = {}
  sampling_idx = get_sampling_idx(
      subset=subset,
      shard=shard,
      subsample=subsample,
      expected_feature_shape=expected_feature_shape,
      pretrained_model=pretrained_model,
      use_euler=use_euler,
      use_mapping=use_mapping,
      split=split)['sampling_idx']
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
                             device=device)
  all_features = []
  all_labels = []
  for idx, blob in tqdm(enumerate(data)):
    if use_mapping:
      frame = f'143_entropy.npy_{idx:06d}'
    else:
      frame = f'{idx}'
    if use_mapping and frame not in sampling_idx:
      print(f'frame {frame} is not in sampling index, skipping')
      continue
    image, label = data_converter_rugd(blob['image'], blob['label'])
    image = torch.from_numpy(image[tf.newaxis].numpy()).to(device)
    
    # run inference
    logits = model(image)['out']
    features = hooks['feat']
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    assert features.shape[1] == expected_feature_shape[0]
    assert features.shape[2] == expected_feature_shape[1]

    label = cv2.resize(label.numpy().squeeze(),
                       dsize=(features.shape[2], features.shape[1]),
                       interpolation=cv2.INTER_NEAREST)
    feature_shape = features.shape[1:3]
    assert label.shape[0] == feature_shape[0]
    assert label.shape[1] == feature_shape[1]
    label = label.flatten()

    features = features.reshape((-1, 256))
    samples = sampling_idx[frame]
    all_features.append(features[samples])
    all_labels.append(label[samples])
    del logits, image, features

  out_features = np.array([[]])
  out_labels = np.array([[]])
  if all_features:
    out_features = np.concatenate(all_features, axis=0)
  else:
    print("No features found, returning empty array")
  if all_labels:
    out_labels = np.concatenate(all_labels, axis=0)
  else:
    print("No labels found, returning empty array")

  return {
      'features': out_features,
      'labels': out_labels,
  }


@memory.cache
def get_dino_embeddings(subset, shard, subsample, device, pretrained_model,
                        expected_feature_shape, use_euler):
  if use_euler:
    data = tfds.load(f'{subset}', split='validation', data_dir=f'{TMPDIR}/datasets')
  else:
    data = tfds.load(f'{subset}', split='validation')
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  sampling_idx = get_sampling_idx(
      subset=subset,
      shard=shard,
      subsample=subsample,
      expected_feature_shape=expected_feature_shape,
      pretrained_model=pretrained_model,
      use_euler=use_euler,
      use_mapping=False)['sampling_idx']
  model = get_dino(device=device)

  all_features = []
  for idx, blob in tqdm(enumerate(data)):
    frame = f'{idx}'
    if frame not in sampling_idx:
      print(f'frame {frame} is not in sampling index, skipping')
      continue
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy())
    image = dino_color_normalize(image).to(device)

    # run inference
    out = model.get_intermediate_layers(image, n=1)[0]
    h = int(image.shape[2] / model.patch_embed.patch_size)
    w = int(image.shape[3] / model.patch_embed.patch_size)
    out = out[:, 1:, :]  # we discard the [CLS] token
    features = out[0].reshape(h, w, out.shape[-1]).permute((2, 0, 1))
    features = torchvision.transforms.functional.resize(
        features,
        size=expected_feature_shape,
        interpolation=PIL.Image.NEAREST)
    features = features.to('cpu').detach().numpy().transpose([1, 2, 0])
    try:
      assert features.shape[0] == expected_feature_shape[0]
      assert features.shape[1] == expected_feature_shape[1]
      assert features.shape[-1] == out.shape[-1]
    except AssertionError:
      raise UserWarning(
          f"shape mismatch! expected {expected_feature_shape}, got {features.shape}"
      )
    # check against deeplab feature size
    features = features.reshape((-1, out.shape[-1]))
    samples = sampling_idx[frame]
    all_features.append(features[samples])
    del image, features

  return {
      'dino_features': np.concatenate(all_features, axis=0),
  }

