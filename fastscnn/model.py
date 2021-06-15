###########################################################################
# Created by: Tramac (https://github.com/Tramac/Fast-SCNN-pytorch)
# Date: 2019-03-25
# Copyright (c) 2017
# License: Apache-2
###########################################################################
"""Fast Segmentation Convolutional Neural Network"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import pi

__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module):

  def __init__(self, num_classes, aux=False, **kwargs):
    super(FastSCNN, self).__init__()
    self.aux = aux
    self.learning_to_downsample = LearningToDownsample(32, 48, 64)
    self.global_feature_extractor = GlobalFeatureExtractor(
        64, [64, 96, 128], 128, 6, [3, 3, 3])
    self.feature_fusion = FeatureFusionModule(64, 128, 128)
    self.classifier = Classifer(128, num_classes)
    if self.aux:
      self.auxlayer = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32), nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Conv2d(32, num_classes, 1))

  def forward(self, x):
    size = x.size()[2:]
    higher_res_features = self.learning_to_downsample(x)
    x = self.global_feature_extractor(higher_res_features)
    x = self.feature_fusion(higher_res_features, x)
    x, feat = self.classifier(x)
    outputs = []
    x = F.interpolate(x, size, mode='bilinear', align_corners=True)
    outputs.append(x)
    if self.aux:
      auxout = self.auxlayer(higher_res_features)
      auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
      outputs.append(auxout)
    outputs.append(feat)
    return tuple(outputs)


class FastSCNNDensity(nn.Module):

  def __init__(self,
               num_classes,
               aux=False,
               n_components=None,
               means=None,
               weights=None,
               covariances=None,
               **kwargs):
    super().__init__()
    self.aux = aux
    self.learning_to_downsample = LearningToDownsample(32, 48, 64)
    self.global_feature_extractor = GlobalFeatureExtractor(
        64, [64, 96, 128], 128, 6, [3, 3, 3])
    self.feature_fusion = FeatureFusionModule(64, 128, 128)
    self.classifier = Classifer(128, num_classes)
    self.gmm = _TorchGMM(128,
                    n_components=n_components,
                    weights=weights,
                    means=means,
                    covariances=covariances)
    # parameters for density estimation
    self.means_ = torch.nn.Parameter(means, requires_grad=False)
    if self.aux:
      self.auxlayer = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32), nn.ReLU(True),
                                    nn.Dropout(0.1),
                                    nn.Conv2d(32, num_classes, 1))

  def forward(self, x):
    size = x.size()[2:]
    higher_res_features = self.learning_to_downsample(x)
    x = self.global_feature_extractor(higher_res_features)
    x = self.feature_fusion(higher_res_features, x)
    x, feat = self.classifier(x)
    outputs = []
    x = F.interpolate(x, size, mode='bilinear', align_corners=True)
    outputs.append(x)
    if self.aux:
      auxout = self.auxlayer(higher_res_features)
      auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
      outputs.append(auxout)
    nll = -self.gmm(feat.permute([0, 2, 3, 1]))
    nll = F.interpolate(nll, size, mode='bilinear', align_corners=True)
    outputs.append(nll)
    return tuple(outputs)


class _TorchGMM(nn.Module):

  def __init__(self, n_features, n_components, means=None, covariances=None, weights=None):
    super().__init__()
    if weights is None:
      weights = torch.rand((n_components,))
    if covariances is None:
      covariances = torch.rand((n_components, n_features, n_features))
    if means is None:
      means = torch.rand((n_components, n_features))
    mix = torch.distributions.Categorical(weights)
    comp  = torch.distributions.MultivariateNormal(means, covariances)
    self.gmm = torch.distributions.MixtureSameFamily(mix, comp)

  def forward(self, x):
    return self.gmm.log_prob(x)


class _GMM(nn.Module):

  def __init__(self,
               n_features,
               n_components=None,
               means=None,
               covariances=None,
               weights=None):
    super().__init__()
    self.n_features = n_features
    self.n_components_init = n_components
    self.means_init = means
    self.var_init = covariances
    self.weights_init = weights
    self._init_params()

  def _init_params(self):
    if self.n_components_init is not None:
      self.n_components = torch.nn.Parameter(torch.tensor(self.n_components_init),
                                             requires_grad=False)
    else:
      self.n_components = torch.nn.Paramerter(torch.tensor(0), requires_grad=False)
    if self.means_init is not None:
      assert self.means_init.size() == (
          1, self.n_components, self.n_features
      ), "Input mu_init does not have required tensor dimensions (1, %i, %i)" % (
          self.n_components, self.n_features)
      # (1, k, d)
      self.means_ = torch.nn.Parameter(self.means_init, requires_grad=False)
    else:
      self.means_ = torch.nn.Parameter(torch.randn(1, self.n_components,
                                                   self.n_features),
                                       requires_grad=False)
    if self.weights_init is not None:
      assert self.weights_init.size() == (
          1, self.n_components, 1
      ), "Input weights do not have required tensor dimensions (1, %i, 1)" % (
          self.n_components)
      # (1, k, d)
      self.weights_ = torch.nn.Parameter(self.weights_init, requires_grad=False)
    else:
      self.weights_ = torch.nn.Parameter(torch.randn(1, self.n_components, 1),
                                         requires_grad=False)
    if self.var_init is not None:
      assert self.var_init.size() == (
          1, self.n_components, self.n_features
      ), "Input var_init does not have required tensor dimensions (1, %i, %i)" % (
          self.n_components, self.n_features)
      # (1, k, d)
      self.var_ = torch.nn.Parameter(self.var_init, requires_grad=False)
    else:
      self.var_ = torch.nn.Parameter(torch.ones(1, self.n_components,
                                                self.n_features),
                                     requires_grad=False)

  def loglikelihood(self, x):
    """
        Computes the log-likelihood of the data under the model.
        args:
            x:                  torch.Tensor (n, 1, d)
        returns:
            per_sample_score:   torch.Tensor (n)
        """
    weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.weights_)
    per_sample_score = torch.logsumexp(weighted_log_prob, dim=-2)
    return per_sample_score[..., 0, 0]

  def check_size(self, x):
    if len(x.size()) == 2:
      # (n, d) --> (n, 1, d)
      x = x.unsqueeze(1)
    return x

  def _estimate_log_prob(self, x):
    """
        Returns a tensor with dimensions (n, k, 1), which indicates the log-likelihood that samples belong to the k-th Gaussian.
        args:
            x:            torch.Tensor (n, d) or (n, 1, d)
        returns:
            log_prob:     torch.Tensor (n, k, 1)
        """
    x = self.check_size(x)
    mu = self.means_
    prec = torch.rsqrt(self.var_)
    log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec**2),
                      dim=-1,
                      keepdim=True)
    log_det = torch.sum(torch.log(prec), dim=-1, keepdim=True)
    return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det


class _ConvBNReLU(nn.Module):
  """Conv-BN-ReLU"""

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=3,
               stride=1,
               padding=0,
               **kwargs):
    super(_ConvBNReLU, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

  def forward(self, x):
    return self.conv(x)


class _DSConv(nn.Module):
  """Depthwise Separable Convolutions"""

  def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
    super(_DSConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(dw_channels,
                  dw_channels,
                  3,
                  stride,
                  1,
                  groups=dw_channels,
                  bias=False), nn.BatchNorm2d(dw_channels), nn.ReLU(True),
        nn.Conv2d(dw_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(True))

  def forward(self, x):
    return self.conv(x)


class _DWConv(nn.Module):

  def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
    super(_DWConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(dw_channels,
                  out_channels,
                  3,
                  stride,
                  1,
                  groups=dw_channels,
                  bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))

  def forward(self, x):
    return self.conv(x)


class LinearBottleneck(nn.Module):
  """LinearBottleneck used in MobileNetV2"""

  def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
    super(LinearBottleneck, self).__init__()
    self.use_shortcut = stride == 1 and in_channels == out_channels
    self.block = nn.Sequential(
        # pw
        _ConvBNReLU(in_channels, in_channels * t, 1),
        # dw
        _DWConv(in_channels * t, in_channels * t, stride),
        # pw-linear
        nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels))

  def forward(self, x):
    out = self.block(x)
    if self.use_shortcut:
      out = x + out
    return out


class PyramidPooling(nn.Module):
  """Pyramid pooling module"""

  def __init__(self, in_channels, out_channels, **kwargs):
    super(PyramidPooling, self).__init__()
    inter_channels = int(in_channels / 4)
    self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
    self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

  def pool(self, x, size):
    avgpool = nn.AdaptiveAvgPool2d(size)
    return avgpool(x)

  def upsample(self, x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)

  def forward(self, x):
    size = x.size()[2:]
    feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
    feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
    feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
    feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
    x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
    x = self.out(x)
    return x


class LearningToDownsample(nn.Module):
  """Learning to downsample module"""

  def __init__(self,
               dw_channels1=32,
               dw_channels2=48,
               out_channels=64,
               **kwargs):
    super(LearningToDownsample, self).__init__()
    self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
    self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
    self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

  def forward(self, x):
    x = self.conv(x)
    x = self.dsconv1(x)
    x = self.dsconv2(x)
    return x


class GlobalFeatureExtractor(nn.Module):
  """Global feature extractor module"""

  def __init__(self,
               in_channels=64,
               block_channels=(64, 96, 128),
               out_channels=128,
               t=6,
               num_blocks=(3, 3, 3),
               **kwargs):
    super(GlobalFeatureExtractor, self).__init__()
    self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels,
                                        block_channels[0], num_blocks[0], t, 2)
    self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0],
                                        block_channels[1], num_blocks[1], t, 2)
    self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1],
                                        block_channels[2], num_blocks[2], t, 1)
    self.ppm = PyramidPooling(block_channels[2], out_channels)

  def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, t, stride))
    for i in range(1, blocks):
      layers.append(block(planes, planes, t, 1))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.bottleneck1(x)
    x = self.bottleneck2(x)
    x = self.bottleneck3(x)
    x = self.ppm(x)
    return x


class FeatureFusionModule(nn.Module):
  """Feature fusion module"""

  def __init__(self,
               highter_in_channels,
               lower_in_channels,
               out_channels,
               scale_factor=4,
               **kwargs):
    super(FeatureFusionModule, self).__init__()
    self.scale_factor = scale_factor
    self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
    self.conv_lower_res = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
    self.conv_higher_res = nn.Sequential(
        nn.Conv2d(highter_in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels))
    self.relu = nn.ReLU(True)

  def forward(self, higher_res_feature, lower_res_feature):
    lower_res_feature = F.interpolate(lower_res_feature,
                                      scale_factor=4,
                                      mode='bilinear',
                                      align_corners=True)
    lower_res_feature = self.dwconv(lower_res_feature)
    lower_res_feature = self.conv_lower_res(lower_res_feature)

    higher_res_feature = self.conv_higher_res(higher_res_feature)
    out = higher_res_feature + lower_res_feature
    return self.relu(out)


class Classifer(nn.Module):
  """Classifer"""

  def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
    super(Classifer, self).__init__()
    self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
    self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
    self.conv = nn.Sequential(nn.Dropout(0.1),
                              nn.Conv2d(dw_channels, num_classes, 1))

  def forward(self, x):
    x = self.dsconv1(x)
    feat = self.dsconv2(x)
    x = self.conv(feat)
    return x, feat


def get_fast_scnn(dataset='citys',
                  pretrained=False,
                  root='./weights',
                  map_cpu=False,
                  **kwargs):
  acronyms = {
      'pascal_voc': 'voc',
      'pascal_aug': 'voc',
      'ade20k': 'ade',
      'coco': 'coco',
      'citys': 'citys',
  }
  from data_loader import datasets
  model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs)
  if pretrained:
    if (map_cpu):
      model.load_state_dict(
          torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset]),
                     map_location='cpu'))
    else:
      model.load_state_dict(
          torch.load(os.path.join(root,
                                  'fast_scnn_%s.pth' % acronyms[dataset])))
  return model
