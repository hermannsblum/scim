import torch
import torchvision
import PIL
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from numpy.random import default_rng
import scipy as sp
import scipy.spatial
import pickle
from tqdm import tqdm
import os
import cv2
from collections import defaultdict
from joblib import Memory
import pickle

from semsegcluster.data.images import convert_img_to_float
from semsegcluster.gdrive import load_gdrive_file
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.sacred_utils import get_observer, get_checkpoint

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


def get_deeplab(pretrained_model, feature_name, device):
  model = torchvision.models.segmentation.deeplabv3_resnet101(
      pretrained=False,
      pretrained_backbone=False,
      progress=True,
      num_classes=40,
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


def get_resnet(feature_name, device):
  model = torchvision.models.resnet101(pretrained=True)
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


@memory.cache
def get_sampling_idx(subset, shard, subsample, pretrained_model,
                     expected_feature_shape):
  rng = default_rng(35643243)
  data = tfds.load(f'scan_net/{subset}', split='validation')
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)

  sampling_idx = {}
  all_voxels = np.array([], dtype=np.int32)
  for blob in tqdm(data.shard(shard, 0)):
    frame = blob['name'].numpy().decode()
    voxel = np.load(os.path.join(
        directory,
        f'{frame}_pseudolabel-voxels.npy')).squeeze().astype(np.int32)
    # interpolate to feature size
    # reshapes to have array <feature_width * feature_height, list of voxels>
    voxel = voxel.reshape(
        (expected_feature_shape[0], voxel.shape[0] // expected_feature_shape[0],
         expected_feature_shape[1],
         voxel.shape[1] // expected_feature_shape[1]))
    voxel = np.swapaxes(voxel, 1, 2).reshape(
        (expected_feature_shape[0] * expected_feature_shape[1], -1))
    # subsampling (because storing all these embeddings would be too much)
    # first subsample from voxels that we have already sampled
    already_sampled = np.logical_and(np.isin(voxel, all_voxels), voxel != 0)
    assert len(already_sampled.shape) == 2
    already_sampled_idx = already_sampled.max(-1)
    if False:
    #if already_sampled_idx.sum() > 0:
      # sample 10% from known voxels
      sampled_idx = np.flatnonzero(already_sampled_idx)
      if sampled_idx.shape[0] > subsample // 10:
        # reduce to a subset
        sampled_idx = rng.choice(sampled_idx,
                                 size=[subsample // 10],
                                 replace=False)
    else:
      sampled_idx = np.array([], dtype=np.int32)
    # now add random samples
    sampled_idx = np.concatenate(
        (sampled_idx,
         rng.choice(expected_feature_shape[0] * expected_feature_shape[1],
                    size=[subsample - sampled_idx.shape[0]],
                    replace=False)),
        axis=0)
    # if a feature corresponds to multiple voxels, favor those that are already sampled
    already_sampled = already_sampled[sampled_idx]
    all_voxels = np.concatenate(
        (all_voxels, voxel[sampled_idx,
                           np.argmax(already_sampled, axis=1)]),
        axis=0)
    sampling_idx[frame] = sampled_idx

  return {
      'voxels': all_voxels,
      'sampling_idx': sampling_idx,
  }


@memory.cache
def get_deeplab_embeddings(subset, shard, subsample, device, pretrained_model,
                           expected_feature_shape, feature_name, pred_name,
                           uncert_name):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  sampling_idx = get_sampling_idx(
      subset=subset,
      shard=shard,
      subsample=subsample,
      expected_feature_shape=expected_feature_shape,
      pretrained_model=pretrained_model)['sampling_idx']
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
                             device=device)

  all_features = []
  all_labels = []
  all_uncertainties = []
  all_entropies = []
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
    if frame not in sampling_idx:
      continue
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
                                  f'{frame}_{uncert_name}.npy')).squeeze()
    # interpolate to feature size
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

  return {
      'features': np.concatenate(all_features, axis=0),
      'prediction': np.concatenate(all_labels, axis=0),
      'uncertainty': np.concatenate(all_uncertainties, axis=0),
      'entropy': np.concatenate(all_entropies, axis=0),
  }


@memory.cache
def get_imagenet_embeddings(subset,
                            shard,
                            subsample,
                            device,
                            pretrained_model,
                            expected_feature_shape,
                            feature_name,
                            interpolate=False):
  data = tfds.load(f'scan_net/{subset}', split='validation')
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'scannet_inference', subset, pretrained_id)
  sampling_idx = get_sampling_idx(
      subset=subset,
      shard=shard,
      subsample=subsample,
      expected_feature_shape=expected_feature_shape,
      pretrained_model=pretrained_model)['sampling_idx']
  model, hooks = get_resnet(feature_name=feature_name, device=device)

  all_features = []
  for blob in tqdm(data):
    frame = blob['name'].numpy().decode()
    if frame not in sampling_idx:
      continue
    image = convert_img_to_float(blob['image'])
    # move channel from last to 2nd
    image = tf.transpose(image, perm=[2, 0, 1])[tf.newaxis]
    image = torch.from_numpy(image.numpy()).to(device)
    # run inference
    _ = model(image)
    features = hooks['feat']
    if interpolate:
      features = torchvision.transforms.functional.resize(
          features,
          size=expected_feature_shape,
          interpolation=PIL.Image.BILINEAR)
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])[0]
    try:
      assert features.shape[0] == expected_feature_shape[0]
      assert features.shape[1] == expected_feature_shape[1]
    except AssertionError:
      raise UserWarning(
          f"shape mismatch! expected {expected_feature_shape}, got {features.shape}"
      )
    # check against deeplab feature size
    features = features.reshape((-1, features.shape[-1]))
    samples = sampling_idx[frame]
    all_features.append(features[samples])
    del image, features

  return {
      'imagenet_features': np.concatenate(all_features, axis=0),
  }
