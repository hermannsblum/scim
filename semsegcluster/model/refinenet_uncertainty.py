import torch

from .refinenet import rf_lw50, rf_lw101, rf_lw152
from .density import TorchGMM, TorchPCA


class RefineNetDensity(torch.nn.Module):

  def __init__(self,
               num_classes,
               size=50,
               n_components=None,
               means=None,
               weights=None,
               covariances=None,
               pca_mean=None,
               pca_components=None,
               feature_layer="mflow_conv_g4_pool",
               **kwargs):
    super().__init__()
    if size == 50:
      self.refinenet = rf_lw50(num_classes, **kwargs)
    elif size == 101:
      self.refinenet = rf_lw101(num_classes, **kwargs)
    elif size == 152:
      self.refinenet = rf_lw152(num_classes, **kwargs)

    # Create hook to get features from intermediate pytorch layer
    self.features = {}

    def get_activation(name, features=self.features):

      def hook(model, input, output):
        features['feat'] = output.detach()

      return hook

    # get feature layer
    feature_layer = getattr(self.refinenet, feature_layer)
    # register hook to get features
    feature_layer.register_forward_hook(get_activation(feature_layer))

    self.pca = TorchPCA(256, 64, mean=pca_mean, components=pca_components)
    self.gmm = TorchGMM(64,
                        n_components=n_components,
                        means=means,
                        weights=weights,
                        covariances=covariances)

  def forward(self, x):
    out = self.refinenet(x)
    features = self.features['feat']
    small_features = self.pca(features.permute([0, 2, 3, 1]))
    nll = -self.gmm(small_features)
    return out, nll
