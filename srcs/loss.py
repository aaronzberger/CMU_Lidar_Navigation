'''
At some point, I will try to implement an Embedding Loss in addition to the 
Classification Loss to try to decrease the difference between training and testing
error.

See (https://arxiv.org/pdf/1712.01511.pdf) for details
'''

import torch
import torch.nn.functional as F

from discriminative import DiscriminativeLoss

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, preds, targets, mask=None): 
        preds = torch.sigmoid(preds)
        
        return F.binary_Fcross_entropy(input=preds, target=targets, weight=None, reduction='mean')