import torch
import torchvision


class DeeplabSML(torch.nn.Module):

  def __init__(self, num_classes, means=None, vars=None, **kwargs):
    super().__init__()
    self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
        num_classes=40, **kwargs)

    if means is None:
      means = torch.rand(num_classes)
    if vars is None:
      sigma = torch.rand(num_classes)
    else:
      sigma = torch.sqrt(vars)

    self.register_buffer('means', means)
    self.register_buffer('sigma', sigma)

  def forward(self, x):
    logits = self.deeplab(x)['out']
    max_logit, c = torch.max(logits, 1)
    sml = -(max_logit - self.means[c]) / self.sigma[c]
    return logits, sml
