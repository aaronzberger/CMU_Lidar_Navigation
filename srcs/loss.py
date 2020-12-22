'''
Simple Cross Entropy Loss, unweighted
'''

import torch
import torch.nn.functional as F
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        return F.binary_cross_entropy(
            input=preds, target=targets, weight=None, reduction='mean')
