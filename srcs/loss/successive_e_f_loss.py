'''
Successive embedding and focal loss to learn tighter
class representations while handling class imbalance

See https://arxiv.org/pdf/1712.01511.pdf
'''

import torch.nn as nn
from embedding_loss import Embedding_Loss
from focal_loss import Focal_Loss


class Successive_E_F_Loss(nn.Module):
    def __init__(self, device, lam, alpha, gamma,
                 margin_s, margin_d, reduction='mean'):
        super(Successive_E_C_Loss, self).__init__()

        # Argument type checking
        if not isinstance(lam, (float, int)) or \
                not isinstance(alpha, (float, int)) or \
                not isinstance(gamma, (float, int)) or \
                not isinstance(margin_s, (float, int)) or \
                not isinstance(margin_d, (float, int)):
            raise TypeError('arguments must be of type (float, int)')

        self.lam = lam

        self.embedding = Embedding_Loss(device, margin_s, margin_d, reduction)
        self.focal = Focal_Loss(device, alpha, gamma, reduction)

    def forward(self, input, target):
        embedding_loss = self.embedding(input, target)
        focal_loss = self.focal(input, target)

        successive_loss = focal_loss + (self.lam * embedding_loss)

        return successive_loss
