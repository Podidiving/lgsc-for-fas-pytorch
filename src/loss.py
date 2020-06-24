import torch
from torch import nn
from torch.nn import functional as F


class TripletLoss(nn.Module):
    """Triplet Loss

    arXiv: https://arxiv.org/pdf/1703.07737.pdf
    """
    def __init__(self, margin=0.5, loss_weight=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        target_ = 1. - target
        bs, fs = pred.shape
        pred = F.normalize(pred, eps=1e-10)
        square_norm = torch.sum(torch.square(pred), 1)

        dist = torch.add(-2.0 * (pred @ pred.T), square_norm.reshape(-1, 1))
        dist = torch.add(dist, square_norm.reshape(1, -1))
        dist[dist < 0] = 0
        dist = torch.sqrt(dist)

        ap_dist = torch.unsqueeze(dist, 2)
        an_dist = torch.unsqueeze(dist, 1)

        loss = ap_dist.repeat(1, 1, bs) - an_dist.repeat(1, bs, 1) + self.margin
        indice_equal = torch.eye(bs, bs)
        indice_not_equal = 1.0 - indice_equal

        broad_matrix = \
            target_.reshape(-1, 1).repeat(1, bs) + \
            target_.reshape(1, -1).repeat(bs, 1)
        pp = \
            torch.eq(broad_matrix, torch.zeros_like(broad_matrix))\
            .type(torch.float32)
        pp = (indice_not_equal * pp).unsqueeze(2)

        pn = \
            torch.eq(broad_matrix, torch.zeros_like(broad_matrix) + 1) \
            .type(torch.float32)
        pn = (indice_not_equal * pn).unsqueeze(0)

        apn = pp.repeat(1, 1, bs) * pn.repeat(bs, 1, 1)
        apn = apn.type(torch.float32)

        loss = loss * apn
        loss[loss < 0] = 0

        num_tri = torch.sum(
            torch.gt(
                loss, torch.zeros_like(loss)
            ).type(torch.float32)
        )
        loss = torch.sum(loss) * self.loss_weight
        loss = loss / (num_tri + 1e-7)

        return loss
