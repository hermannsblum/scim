from sacred import Experiment
from joblib import Memory
import numpy as np
import torch
import cv2
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
tf.config.set_visible_devices([], 'GPU')
import os
import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)
from gmmtorch.gmm import GaussianMixture
from semsegcluster.sacred_utils import get_observer, get_checkpoint
from semsegcluster.settings import TMPDIR, EXP_OUT
from semsegcluster.data.tfds_to_torch import TFDataIterableDataset

from deeplab.oaisys_sampling import get_deeplab, get_sampling_idx
from deeplab.oaisys_utils import data_converter_rugd

ex = Experiment()
ex.observers.append(get_observer())

memory = Memory("/tmp")

ex.add_config(
    subsample=100,
    shard=5,
    device='cuda',
    feature_name='classifier.2',
    expected_feature_shape=[60, 80],
    pred_name='pred',
    uncert_name='maxlogit-pp',
    ignore_other=True,
    apply_scaling=True,
    same_voxel_close=None,
    distance_activation_factor=None,
    normalize=False,
    use_euler=False,
    training_set='oaisys16k_rugd',
    subset='oaisys_trajectory',
    n_components=11,
    split='train',
    num_classes=15,
    batch_size=1,
)

@ex.command
def gmm_predict(
    _run,
    pretrained_model,
    device,
    training_set,
    subset,
    use_euler,
    feature_name,
    subsample,
    num_classes,
    normalize,
    expected_feature_shape,
):
  # setup euler
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{subset}.tar')

  # load gmm model parameters
  _, pretrained_id = get_checkpoint(pretrained_model)
  training_directory = os.path.join(EXP_OUT, 'oaisys_inference', training_set, pretrained_id)
  mu =  torch.tensor(np.load(os.path.join(training_directory, f'gmm_mu_{subsample}_{num_classes}.npy')))
  var =  torch.tensor(np.load(os.path.join(training_directory, f'gmm_var_{subsample}_{num_classes}.npy')))
  pi =  torch.tensor(np.load(os.path.join(training_directory, f'gmm_pi_{subsample}_{num_classes}.npy')))

  # setup gmm model
  gmm_model = GaussianMixture(n_components=mu.size(1), n_features=mu.size(2), covariance_type='diag', mu_init=mu, var_init=var, pi_init=pi)


  # setup data
  if use_euler:
    data = tfds.load(f'{subset}', split='validation',  as_supervised=True, data_dir=f'{TMPDIR}/datasets')
  else:
    data = tfds.load(f'{subset}', split='validation', as_supervised=True)
  dataset = TFDataIterableDataset(data.map(data_converter_rugd))
  data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=1,
                                           pin_memory=True,
                                           drop_last=True)

  # setup deeplab
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
                             device=device,
                             num_classes=num_classes)
  

  # setup output directory
  directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{subset}', pretrained_id)
  os.makedirs(directory, exist_ok=True)

  # INFERENCE
  for idx, (image, _) in enumerate(tqdm(data_loader)):
    # run inference
    print(image.shape)
    logits = model(image.to(device))['out']

    # get features
    features = hooks['feat']
    if normalize:
      features = torch.nn.functional.normalize(features, dim=1)
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    features = features.reshape((-1, 256))
    features = torch.tensor(features)
    # predict gmm
    gmm_pred = gmm_model.predict(features).detach().to('cpu').numpy()
    gmm_pred = gmm_pred.reshape((expected_feature_shape[0], expected_feature_shape[1]))
    gmm_pred = cv2.resize(gmm_pred, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    np.save(os.path.join(directory, f'{idx:06d}_gmm_pred_{subsample}_{num_classes}.npy'),
            gmm_pred)
    

    # predict max log prob
    gmm_max_log_prob = gmm_model.predict_max_log_prob(features).detach().to('cpu').numpy()
    gmm_max_log_prob = gmm_max_log_prob.reshape((expected_feature_shape[0], expected_feature_shape[1]))
    gmm_max_log_prob = cv2.resize(gmm_max_log_prob, (640, 480),
                              interpolation=cv2.INTER_NEAREST)
    # save output
    np.save(os.path.join(directory, f'{idx:06d}_gmm_max_log_prob_{subsample}_{num_classes}.npy'),
            gmm_max_log_prob)
    


@ex.main
def gmm_fit(
    _run,
    pretrained_model,
    device,
    subset,
    use_euler,
    feature_name,
    shard,
    subsample,
    expected_feature_shape,
    n_components,
    split,
    num_classes,
    batch_size,
    normalize,
):
  # setup euler data
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{subset}.tar')
  
  # deeplab setup
  model, hooks = get_deeplab(pretrained_model=pretrained_model,
                             feature_name=feature_name,
                             device=device,
                             num_classes=num_classes)
  # load data
  if use_euler:
    data = tfds.load(f'{subset}', split=split,  as_supervised=True, data_dir=f'{TMPDIR}/datasets')
  else:
    data = tfds.load(f'{subset}', split=split, as_supervised=True)
  dataset = TFDataIterableDataset(data.map(data_converter_rugd))
  data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           pin_memory=True,
                                           drop_last=True)
  # get sampling idx
  sampling_idx = get_sampling_idx(
      subset=subset,
      shard=shard,
      subsample=subsample,
      expected_feature_shape=expected_feature_shape,
      pretrained_model=pretrained_model,
      use_euler=use_euler,
      use_mapping=False,
      split=split)['sampling_idx']
  
  # get all features
  all_features = []

  for idx, (image, _) in tqdm(enumerate(data_loader)):
    frame= f'{idx}'
    # run inference
    logits = model(image.to(device))['out']
    # get features
    features = hooks['feat']
    if normalize:
      features = torch.nn.functional.normalize(features, dim=1)
    features = features.to('cpu').detach().numpy().transpose([0, 2, 3, 1])
    assert features.shape[-1] == 256
    features = features.reshape((-1, 256))
    # get predictions
    logits = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    _, pred = torch.max(logits, 1)
    # get features for sampled pixels
    samples = sampling_idx[frame]
    out_features = features[samples]
    all_features.append(out_features)

  # initialise gmm
  gmm_model = GaussianMixture(n_components=n_components, n_features=256, covariance_type='diag')

  # fit gmm
  features = torch.tensor(np.concatenate(all_features, axis=0))
  gmm_model.fit(features)

  # save model parameters
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'oaisys_inference', f'{subset}', pretrained_id)
  np.save(os.path.join(directory, f'gmm_mu_{subsample}_{num_classes}.npy'),
            gmm_model.mu.numpy())
  np.save(os.path.join(directory, f'gmm_var_{subsample}_{num_classes}.npy'),
            gmm_model.var.numpy())
  np.save(os.path.join(directory, f'gmm_pi_{subsample}_{num_classes}.npy'),
            gmm_model.pi.numpy())


  return gmm_model


if __name__ == '__main__':
  ex.run_commandline()