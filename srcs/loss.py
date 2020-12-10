import torch
import torch.nn as nn
import torch.nn.functional as F

from discriminative import DiscriminativeLoss


# from config import reg_channels, cls_channels, output_channels, alpha, beta

class ClassificationLoss(nn.Module):
    def __init__(self, device, config, num_classes=1):
        super(ClassificationLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        # self.alpha = config['alpha']
        self.beta = config['beta']

        self.discriminative_loss = DiscriminativeLoss(0.5, 3.0, 2)

    def focal_loss(self, x, y, mask=None):
        '''Focal loss.
        Args:
          x: (tensor) sized [BatchSize, Height, Width].
          y: (tensor) sized [BatchSize, Height, Width].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.9
        gamma = 0

        log_x = F.logsigmoid(x)
        x = torch.sigmoid(x)
        x_t = x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                        # x_t = 1 -x  if label = 0

        log_x_t = log_x * (2 * y - 1) + (1 - y) # x_t = x     if label = 1
                                        # x_t = 1 -x  if label = 0

        alpha_t = torch.ones_like(x_t) * alpha
        alpha_t = alpha_t * (2 * y - 1) + (1 - y)

        loss = -alpha_t * (1-x_t)**gamma * log_x_t

        if mask is not None:
            loss = loss * mask

        return loss.mean()

    def cross_entropy(self, x, y, weight=None):
        #print(x.size(), y.size(), weight.size())
        #print(x.dtype, y.dtype, weight.dtype)
        # raise ValueError()

        # Deprecation error on Oct 26, 2020
#         x = F.sigmoid(x)
        x = torch.sigmoid(x)

        # Deprecation error on Oct 26, 2020
#         return F.binary_cross_entropy(input=x, target=y, weight=weight, reduction='elementwise_mean')
        return F.binary_cross_entropy(input=x, target=y, weight=weight, reduction='mean')


    def forward(self, preds, targets, mask=None):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          preds: (tensor)  cls_preds + reg_preds, sized[batch_size, height, width, 7]
          cls_preds: (tensor) predicted class confidences, sized [batch_size, height, width, 1].
          cls_targets: (tensor) encoded target labels, sized [batch_size, height, width, 1].
          loc_preds: (tensor) predicted target locations, sized [batch_size, height, width, 6 or 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, height, width, 6 or 8].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        loss = self.cross_entropy(preds, targets, weight=mask)
        
        return loss

class EmbeddingLoss(nn.Module):
    def __init__(self, device, config):
        super(CustomLoss, self).__init__()

        self.device = device
        self.discriminative_loss = DiscriminativeLoss(0.5, 3.0, 2)
        self.max_num_instances = config['max_num_instances']

    def forward(self, preds, targets, n_instances):
        return self.discriminative_loss(preds, targets, n_instances,
                self.max_num_instances)

def test():
    from utils import load_config
    config, _, _, _ = load_config('default')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    loss = CustomLoss(device, config)

    pred = torch.sigmoid(torch.randn(1, 5, 2, 2))
    label = torch.tensor([[[[1, 0.4, 0.5, 0.5, 1], [0, 0.2, 0.5, 0.5, 1]], [[0, 0.1,
        0.1, 0.1, 1], [1, 0.8, 0.4, 0.4, 1]]]]).permute(0, 3, 2, 1)
    loss = loss(pred, label)
    print(loss)


if __name__ == '__main__':
    test()
