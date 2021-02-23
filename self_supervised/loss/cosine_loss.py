import torch
import torch.nn.functional as F


class CosineLoss(torch.nn.Module):
    r"""Cosine loss.

    .. note::

        Also known as normalized L2 distance.
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        outputs = F.normalize(outputs, dim=-1, p=2)
        targets = F.normalize(targets, dim=-1, p=2)
        return (2 - 2 * (outputs * targets).sum(dim=-1)).mean()
