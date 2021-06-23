import torch

from .refinenet import rf_lw50, rf_lw101, rf_lw152
from .gmm import TorchGMM


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
               **kwargs):
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
          features[name] = output.detach()

        return hook

      # get feature layer
      feature_layer = getattr(self.base_model, feature_layer_name)
      # register hook to get features
      feature_layer.register_forward_hook(get_activation(feature_layer_name))
