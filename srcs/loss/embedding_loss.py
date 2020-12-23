'''
Embedding Loss to decrease difference between training & testing loss

See https://arxiv.org/pdf/1712.01511.pdf
'''

import torch
import torch.nn as nn
from config import exp_name
from utils import load_config


class Embedding_Loss(nn.Module):
    def __init__(self, device, margin_similar=0.0, margin_dissimilar=2.0,
                 reduction='mean', contrastive=True):
        super(Embedding_Loss, self).__init__()

        # Argument type checking
        if reduction not in ['mean', 'sum', 'none']:
            raise NotImplementedError(
                'Reduction {} not implemented.').format(reduction)
        if not isinstance(margin_similar, (float, int)) or \
                not isinstance(margin_dissimilar, (float, int)):
            raise TypeError('arguments must be of type (float, int)')

        self.contrastive = contrastive
        self.reduction = reduction
        self.device = device

        # Pull pixels of same label closer than this
        self.margin_similar = (torch.tensor([margin_similar])).to(device)

        # Push pixels of diff labels further than this
        self.margin_dissimilar = (torch.tensor([margin_dissimilar])).to(device)

        _, _, self.batch_size, _ = load_config(exp_name)

    def contrastive_loss(self, input, target):
        # Flatten the predictions and truth
        x = input.view(-1, 1)
        y = target.view(-1, 1)

        # Split into a and b groups for comparison
        x_a, x_b = torch.split(x, len(x) / 2)
        y_a, y_b = torch.split(y, len(y) / 2)

        squared_norm = torch.pow(torch.norm(x_a - x_b), 2)

        loss_sim = squared_norm - self.margin_similar
        loss_dis = self.margin_dissimilar - squared_norm

        contrastive_loss = torch.where(
            y_a == y_b,
            loss_sim,
            loss_dis
        )

        contrastive_loss[contrastive_loss < 0] = 0

        return contrastive_loss

    def forward(self, input, target):
        if self.contrastive:
            embedding_loss = self.contrastive_loss(input, target)

        if self.reduction == 'mean':
            return embedding_loss.mean()
        elif self.reduction == 'sum':
            return embedding_loss.sum()
        else:
            return embedding_loss
