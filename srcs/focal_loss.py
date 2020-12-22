'''
Focal Loss to handle class imbalance

See https://arxiv.org/abs/1708.02002
'''

import torch
import torch.nn.functional as F
import torch.nn as nn


class Focal_Loss(nn.Module):
    def __init__(self, device, alpha=0.75, gamma=2, reduction='mean'):
        super(Focal_Loss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise NotImplementedError(
                'Reduction {} not implemented.').format(reduction)
        if not isinstance(alpha, (float, int)):
            raise TypeError('alpha must be of float type')
        if not isinstance(gamma, (float, int)):
            raise TypeError('gamma must of float type')

        # BCE must take a tensor (in case of more than one class)
        # Also must be on the same device as the input and target
        self.alpha = (torch.tensor([alpha])).to(device)

        # The higher the value of gamma, the greater the weighting of
        # hard-to-classifly pixels
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Inputs must be [0:1] for BCE input
        input = torch.sigmoid(input)

        ce_loss = F.binary_cross_entropy(
            input, target, weight=None, reduction='none')

        # BCE = -log(pt) so pt = e^-BCE
        pt = torch.exp(-ce_loss)

        # Alpha helps with class imbalance as well
        alpha_tensor = torch.where(target == 1, self.alpha, 1 - self.alpha)

        focal_loss = alpha_tensor * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
