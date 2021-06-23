import torch


class TorchGMM(nn.Module):
  """GMM for inference in torch."""

  def __init__(self,
               n_features,
               n_components,
               means=None,
               covariances=None,
               weights=None,
               covariance_type="tied",
               reg_covar=1e-6):
    super().__init__()

    if weights is None:
      weights = torch.rand((n_components,))
    if covariances is None:
      covariances = torch.tile(torch.unsqueeze(torch.eye(n_features), 0),
                               (n_components, 1, 1))
    if means is None:
      means = torch.rand((n_components, n_features))

    if torch.cuda.is_available():
      means = means.cuda()
      covariances = covariances.cuda()
      weights = weights.cuda()

    self.register_buffer('means', means)
    self.register_buffer('covariances', covariances)
    self.register_buffer('weights', weights)


    mix = torch.distributions.Categorical(self.weights)
    comp = torch.distributions.MultivariateNormal(self.means, self.covariances)
    self.gmm = torch.distributions.MixtureSameFamily(mix, comp)

  def _load_from_state_dict(self, *args, **kwargs):
    super(_TorchGMM, self)._load_from_state_dict(*args, **kwargs)
    # Ugly fix to make sure distributions can be loaded -> recreate distributions
    mix = torch.distributions.Categorical(self.weights)
    comp = torch.distributions.MultivariateNormal(self.means, self.covariances)
    self.gmm = torch.distributions.MixtureSameFamily(mix, comp)

  def forward(self, x):
    return torch.unsqueeze(self.gmm.log_prob(x), 3)
