import torch

from .refinenet import rf_lw50, rf_lw101, rf_lw152


class RefineNetSML(torch.nn.Module):

  def __init__(self,
               num_classes,
               size=50,
               means=None,
               vars=None,
               **kwargs):
    super().__init__()
    if size == 50:
      self.refinenet = rf_lw50(num_classes, **kwargs)
    elif size == 101:
      self.refinenet = rf_lw101(num_classes, **kwargs)
    elif size == 152:
      self.refinenet = rf_lw152(num_classes, **kwargs)

    if means is None:
      means = torch.rand(num_classes)
    if vars is None:
      sigma = torch.rand(num_classes)
    else:
      sigma = torch.sqrt(vars)

    self.register_buffer('means', means)
    self.register_buffer('sigma', sigma)

  def forward(self, x):
    logits = self.refinenet(x)
    max_logit, c = torch.max(logits, 1)
    sml = -(max_logit - self.means[c]) / self.sigma[c]
    return logits, sml
