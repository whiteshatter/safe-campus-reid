# encoding: utf-8
"""
a lot copy from reid-strong-baseline
"""
import os
import numpy as np
import torch.nn.functional as F
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import torch
import torch.nn as nn
import math
from .triplet import TripletLoss as LAGNetTripletLoss
from .center_loss import CenterLoss


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

def hard_id_mining(dist_mat, labels):
    assert len(dist_mat.size()) == 2
    M = dist_mat.size(0)
    N = dist_mat.size(1)

    # shape [M]
    pos_mask = torch.eye(N, device=labels.device)[labels].type(torch.bool)
    # shape [M*(N-1)]
    neg_mask = ~pos_mask

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap = dist_mat.masked_select(pos_mask)
    dist_an = dist_mat.masked_select(neg_mask).view(M, N-1).min(1)[0]

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return dict(loss=loss, dist_ap=dist_ap.mean(), dist_an=dist_an.mean())

class TripletIDLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, ranking_loss=True, id_loss=False):
        self.margin = margin
        if ranking_loss:
            if margin is not None:
                self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
            else:
                self.ranking_loss = nn.SoftMarginLoss(reduction='none')
        else:
            self.ranking_loss = None
        self.id_loss = id_loss

    def __call__(self, global_feat, labels, id_feature_dict, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, id_feature_dict)
        dist_ap, dist_an = hard_id_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.ranking_loss is not None:
            if self.margin is not None:
                loss = self.ranking_loss(dist_an, dist_ap, y)
            else:
                loss = self.ranking_loss(dist_an - dist_ap, y)
        else:
            loss = dist_ap - dist_an

        if self.id_loss:
            mask = loss>0
            loss1 = loss[mask].sum()
            loss2 = dist_ap[mask].sum()
            loss = (loss1+loss2)/loss.shape[0]
        else:
            loss = loss.mean()

        return dict(id_loss=loss, id_ap=dist_ap.mean(), id_an=dist_an.mean())


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class LAGNetLoss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(LAGNetLoss, self).__init__()
        # print('[INFO] Making loss...')
        self.last_loss = sys.float_info.max
        self.nGPU = args.nGPU
        self.args = args
        self.loss = []
        self.loss_module = nn.ModuleList()
        
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            elif loss_type == 'Triplet':
                loss_function = LAGNetTripletLoss(args.margin)
            elif loss_type == 'L2':
                loss_function = CenterLoss()

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
                })
            

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {} {}'.format(l['weight'], l['type'], l))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        #self.feature_center = torch.zeros(args.num_classes, args.num_attentions * args.num_features).to(self.device)
        
        if not args.cpu and args.nGPU > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.nGPU)
            )

    def forward(self, feature_center_batch, outputs, labels):
        losses = []

        for i, l in enumerate(self.loss):
            if self.args.model == 'LAG' and l['type'] == 'Triplet':
                loss = [l['function'](output, labels) for output in outputs[2:7]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif self.args.model == 'LAG' and l['type'] == 'L2':
                loss = l['function'](outputs[1], feature_center_batch)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif self.args.model == 'LAG' and l['function'] is not None:
                loss = [l['function'](output, labels) for output in outputs[7:]]
                loss = sum(loss) / len(loss)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            else:
                pass
        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, batches):
        self.log[-1].div_(batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.jpg'.format(apath, l['type']))
            plt.close(fig)

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.nGPU == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))
        #torch.save(self.feature_center.cpu(), os.path.join(apath, 'feature_center.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        #self.feature_center = torch.load(os.path.join(apath, 'feature_center.pt')).to(self.device)
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

