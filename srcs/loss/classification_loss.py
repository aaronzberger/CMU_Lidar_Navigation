'''
Simple Cross Entropy Loss, unweighted
'''

import torch
import torch.nn.functional as F
import torch.nn as nn


class Classification_Loss(nn.Module):
    def __init__(self, alpha, reduction='mean'):
        super(Classification_Loss, self).__init__()

        # Argument checking
        if reduction not in ['mean', 'sum', 'none']:
            raise NotImplementedError(
                'Reduction {} not implemented.').format(reduction)

        self.reduction = reduction
        self.alpha = alpha

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        # Alpha for class imbalance
        alpha = (targets * self.alpha) + ((1 - self.alpha) * (1 - targets))

        return F.binary_cross_entropy(
            input=preds, target=targets,
            weight=alpha, reduction=self.reduction)
