from torch.autograd import Variable
import torch
import torch.nn as nn


class Discriminative_Loss(nn.Module):

    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 reduction='mean'):
        super(Discriminative_Loss, self).__init__()

        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if self.norm not in [1, 2]:
            raise ValueError("Norm must be in [1, 2], meaning L1 or L2")

    def forward(self, preds, target, n_clusters):
        with torch.no_grad():
            return self.discriminative_loss(preds, target, n_clusters)

    def discriminative_loss(self, preds, target, n_clusters):
        batch_size, num_instances, height, width = preds.size()
        max_n_clusters = target.size(1)

        # [BS, C, H, W] -> [BS, C, H*W]
        preds = preds.contiguous().view(
            batch_size, num_instances, height * width)
        target = target.contiguous().view(
            batch_size, max_n_clusters, height * width)

        c_means = self.cluster_means(preds, target, n_clusters)
        l_var = self.variance_term(preds, target, c_means, n_clusters)
        l_dist = self.distance_term(c_means, n_clusters)
        l_reg = self.regularization_term(c_means, n_clusters)
        print('c_means', c_means)
        # print('var_term', l_var)
        # print('dist_term', l_dist)
        # print('reg_term', l_reg)

        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

        return loss

    def cluster_means(self, preds, target, max_num_instances):
        batch_size, pred_num_instances, pred_map = preds.size()
        truth_num_instances = target.size(1)

        preds = preds.unsqueeze(2).expand(
            batch_size, pred_num_instances, truth_num_instances, pred_map)
        # batch_size, 1, max_num_instances, truth_pred_map
        target = target.unsqueeze(1)
        # batch_size, num_instances, max_num_instances, pred_map
        preds = preds * target

        means = []
        for i in range(batch_size):
            # n_features, n_clusters, n_loc
            input_sample = preds[i, :, :max_num_instances[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :, :max_num_instances[i]]
            # n_features, n_cluster
            mean_sample = input_sample.sum(2) / target_sample.sum(2)

            # padding
            n_pad_clusters = truth_num_instances - max_num_instances[i]
            assert n_pad_clusters >= 0
            if n_pad_clusters > 0:
                pad_sample = torch.zeros(pred_num_instances, n_pad_clusters)
                pad_sample = Variable(pad_sample)
                pad_sample = pad_sample.cuda()
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)

        return means

    def variance_term(self, preds, target, c_means, n_clusters):
        batch_size, pred_num_instances, n_loc = preds.size()
        truth_num_instances = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(
            batch_size, pred_num_instances, truth_num_instances, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        preds = preds.unsqueeze(2).expand(
            batch_size, pred_num_instances, truth_num_instances, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((preds - c_means), self.norm, 1) -
                           self.delta_var, min=0) ** 2) * target

        var_term = 0
        for i in range(batch_size):
            # n_clusters, n_loc
            var_sample = var[i, :n_clusters[i]]
            # n_clusters, n_loc
            target_sample = target[i, :n_clusters[i]]

            # n_clusters
            c_var = var_sample.sum(1) / target_sample.sum(1)
            var_term += c_var.sum() / n_clusters[i]
        var_term /= batch_size

        return var_term

    def distance_term(self, c_means, n_clusters):
        batch_size, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(batch_size):
            if n_clusters[i] <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            margin = Variable(margin)
            margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term /= batch_size

        return dist_term

    def regularization_term(self, c_means, n_clusters):
        batch_size, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(batch_size):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= batch_size

        return reg_term
