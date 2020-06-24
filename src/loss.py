from torch import nn
from utils import batch_all_triplet_loss


class TripletLoss(nn.Module):
    """Triplet Loss

    arXiv: https://arxiv.org/pdf/1703.07737.pdf
    """

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        loss, _ = batch_all_triplet_loss(target, pred, self.margin, squared=True)
        return loss
