'''
Embedding Loss to decrease difference between training & testing loss

See https://arxiv.org/pdf/1712.01511.pdf
'''

import torch
import torch.nn as nn


class Embedding_Loss(nn.Module):
    def __init__(self, device, margin_similar=0.0, margin_dissimilar=1.0,
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

        self.margin_similar = (torch.tensor([margin_similar])).to(device)

        self.margin_dissimilar = (torch.tensor([margin_dissimilar])).to(device)

    def contrastive_loss(self, input, target):
        if not input.shape[0] >= 2:
            raise ValueError('Contrastive Embedding Loss requires two images \
                to compare. Increase batch size to >= 2')

        x = input.view(-1, 1)
        y = target.view(-1, 1)

        # Split into a and b groups for comparison
        mid_index = len(x) / 2
        x_a, x_b = x[:mid_index], x[mid_index:]
        y_a, y_b = y[:mid_index], y[mid_index:]

        # Euclidian distance between all points
        point_dists = (x_a - x_b).abs()

        # Pull points of the same label closer than margin_similar
        # and points of opposite labels farther than margin_dissimilar
        contrastive_loss = torch.where(
            y_a == y_b,
            point_dists,
            1 - point_dists
        )

        either_ones = (y_a + y_b) > 0

        alpha = torch.ones_like(x_a) * 0.01
        alpha[either_ones] = 1.0

        # Negative loss means it's within the margin, so set to 0
        contrastive_loss[contrastive_loss < 0] = 0

        return alpha * contrastive_loss

    def forward(self, input, target):
        if self.contrastive:
            embedding_loss = self.contrastive_loss(input, target)

        if self.reduction == 'mean':
            return embedding_loss.mean()
        elif self.reduction == 'sum':
            return embedding_loss.sum()
        else:
            return embedding_loss
