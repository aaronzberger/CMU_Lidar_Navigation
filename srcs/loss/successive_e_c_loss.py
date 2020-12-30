'''
Successive embedding and alpha-balanced BCE loss to learn tighter
class representations while handling class imbalance

See https://arxiv.org/pdf/1712.01511.pdf
'''

import torch.nn as nn
from embedding_loss import Embedding_Loss
from classification_loss import Classification_Loss


class Successive_E_C_Loss(nn.Module):
    def __init__(self, device, lam, alpha,
                 margin_s, margin_d, reduction='mean'):
        super(Successive_E_C_Loss, self).__init__()

        # Argument type checking
        if not isinstance(lam, (float, int)) or \
                not isinstance(alpha, (float, int)) or \
                not isinstance(margin_s, (float, int)) or \
                not isinstance(margin_d, (float, int)):
            raise TypeError('arguments must be of type (float, int)')

        self.lam = lam

        self.embedding = Embedding_Loss(device, margin_s, margin_d, reduction)
        self.classification = Classification_Loss(alpha, reduction)

    def forward(self, input, target):
        embedding_loss = self.embedding(input, target)
        classification_loss = self.classification(input, target)

        successive_loss = classification_loss + (self.lam * embedding_loss)

        return successive_loss, embedding_loss * self.lam, classification_loss
