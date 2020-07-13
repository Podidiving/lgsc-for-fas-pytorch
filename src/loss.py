import torch
from torch import nn


# source code from: https://github.com/Yuol96/pytorch-triplet-loss
def pairwise_distances(embeddings, squared=False):
    """
    ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
    """
    # get dot product (batch_size, batch_size)
    dot_product = embeddings.mm(embeddings.t())

    # a vector
    square_sum = dot_product.diag()

    distances = (
        square_sum.unsqueeze(1) - 2 * dot_product + square_sum.unsqueeze(0)
    )

    distances = distances.clamp(min=0)

    if not squared:
        epsilon = 1e-16
        mask = torch.eq(distances, 0).float()
        distances += mask * epsilon
        distances = torch.sqrt(distances)
        distances *= 1 - mask

    return distances


# source code from: https://github.com/Yuol96/pytorch-triplet-loss
def get_valid_triplets_mask(labels):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = ~indices_equal
    i_ne_j = indices_not_equal.unsqueeze(2)
    i_ne_k = indices_not_equal.unsqueeze(1)
    j_ne_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_ne_j & i_ne_k & j_ne_k

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    i_eq_j = label_equal.unsqueeze(2)
    i_eq_k = label_equal.unsqueeze(1)
    i_ne_k = ~i_eq_k
    valid_labels = i_eq_j & i_ne_k

    mask = distinct_indices & valid_labels
    return mask


# source code from: https://github.com/Yuol96/pytorch-triplet-loss
def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    get triplet loss for all valid triplets and average over those triplets whose loss is positive.
    """

    distances = pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = distances.unsqueeze(2)
    anchor_negative_dist = distances.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # get a 3D mask to filter out invalid triplets
    mask = get_valid_triplets_mask(labels)

    triplet_loss = triplet_loss * mask.float()
    triplet_loss.clamp_(min=0)

    # count the number of positive triplets
    epsilon = 1e-16
    num_positive_triplets = (triplet_loss > 0).float().sum()
    num_valid_triplets = mask.float().sum()
    fraction_positive_triplets = num_positive_triplets / (
        num_valid_triplets + epsilon
    )

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

    return triplet_loss, fraction_positive_triplets


class TripletLoss(nn.Module):
    """Triplet Loss

    arXiv: https://arxiv.org/pdf/1703.07737.pdf
    """

    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        loss, _ = batch_all_triplet_loss(
            target, pred, self.margin, squared=True
        )
        return loss
