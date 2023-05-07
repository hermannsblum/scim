from sacred import Experiment
from joblib import Memory
import numpy as np
import torch
import tensorflow_datasets as tfds
import tensorflow as tf
import sklearn
import sklearn.cluster
import sklearn.preprocessing
import sklearn.metrics
tf.config.set_visible_devices([], 'GPU')
import os
import sys
par_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(par_dir)
from gmmtorch.gmm import GaussianMixture
from semsegcluster.sacred_utils import get_observer, get_checkpoint
from semsegcluster.settings import TMPDIR, EXP_OUT

from deeplab.oaisys_sampling import get_deeplab_embeddings_features_labels
ex = Experiment()
ex.observers.append(get_observer())


memory = Memory("/tmp")

ex.add_config(
    subsample=100,
    shard=5,
    device='cuda',
    feature_name='classifier.2',
    expected_feature_shape=[60, 80],
    apply_scaling=True,
    same_voxel_close=None,
    uncertainty_threshold=-3,
    distance_activation_factor=None,
    normalize=True,
    use_euler=False,
    subset='oaisys_trajectory',
    n_components=11,
    split='train',
)

@ex.main
def gmm_test(
    _run,
    pretrained_model,
    device,
    subset,
    use_euler,
    feature_name,
    shard,
    subsample,
    expected_feature_shape,
    normalize,
    n_components,
    split,
):
  if use_euler:
    os.system(f'mkdir {TMPDIR}/datasets')
    os.system(f'tar -C {TMPDIR}/datasets -xvf /cluster/project/cvg/students/loewsi/datasets/{subset}.tar')
  # out = get_deeplab_embeddings_features_labels(subset=subset,
  #                            shard=shard,
  #                            subsample=subsample,
  #                            device=device,
  #                            pretrained_model=pretrained_model,
  #                            expected_feature_shape=expected_feature_shape,
  #                            feature_name=feature_name,
  #                            use_euler=use_euler,
  #                            use_mapping=False,
  #                            split=split)
  # print(out['features'])
  # print(out['features'].shape)
  # print(out['labels'])
  # print(out['labels'].shape)

  # return
  # if normalize:
  #   out['features'] = sklearn.preprocessing.normalize(out['features'])

  # features = torch.tensor(out['features'])
  # print(features)
  # print(features.size(1))
  _, pretrained_id = get_checkpoint(pretrained_model)
  directory = os.path.join(EXP_OUT, 'oaisys_inference', subset, pretrained_id)
  mu =  torch.tensor(np.load(os.path.join(directory, 'gmm_mu.npy')))
  var =  torch.tensor(np.load(os.path.join(directory, 'gmm_var.npy')))
  pi =  torch.tensor(np.load(os.path.join(directory, 'gmm_pi.npy')))
  print('mu: ', mu)
  print('var: ', var)
  print('pi: ', pi)
  model = GaussianMixture(n_components=mu.size(1), n_features=mu.size(2), covariance_type='diag', mu_init=mu, var_init=var, pi_init=pi)
  # model.fit(features)

  print('model: ', model)
  print('mu: ', model.mu)
  print('var: ', model.var)
  print('pi: ', model.pi)

  return 


if __name__ == '__main__':
  ex.run_commandline()