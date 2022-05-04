import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        losses = 0.5 * (label.float() * euclidean_distance + (1 + (-1 * label) ).float() * F.relu(self.margin - (euclidean_distance + self.eps).sqrt()).pow(2))
        loss_contrastive = torch.mean(losses)

        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """
    @staticmethod
    def add_criterion_specific_args(parent_parser):
        """Triplet Loss parameters
        Args:
            margin: margin of triplet loss
        """
        parser = parent_parser.add_argument_group("triplet loss params")
        parser.add_argument("--training.triplet.margin", type=float, default=1.0)
        return parent_parser

    def __init__(self, margin=1.):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True) -> torch.float:
        # anchor.shape == positive.shape == negative.shape == [batch_size, feat_embedding]
        distance_positive = F.cosine_similarity(anchor,positive) # maximize 
        distance_negative = F.cosine_similarity(anchor,negative)  # minimize (can take to the power of `.pow(.5)`)
        losses = (1- distance_positive)**2 + (0 - distance_negative)**2      #Margin not used in cosine case. 
        return losses.mean() if size_average else losses.sum()


class QuadrupletLoss(nn.Module):
    @staticmethod
    def add_criterion_specific_args(parent_parser):
        """Triplet Loss parameters
        Args:
            margin: margin of triplet loss
        """
        parser = parent_parser.add_argument_group("quadruplet loss params")
        parser.add_argument("--training.quadruplet.margin_alpha", type=float, default=.1)
        parser.add_argument("--training.quadruplet.margin_beta", type=float, default=.01)
        return parent_parser
    
    def __init__(self, margin_alpha: float=.1, margin_beta: float=.01, **kwargs):
        super(QuadrupletLoss, self).__init__()
        self.margin_a = margin_alpha
        self.margin_b = margin_beta

    def forward(self, ap_dist, an_dist, nn_dist) -> torch.float:
        # ap_dist.shape == an_dist.shape == nn_dist.shape == torch.Size([batch_size])
        ap_dist2 = torch.square(ap_dist)
        an_dist2 = torch.square(an_dist)
        nn_dist2 = torch.square(nn_dist)

        diff1 = ap_dist2-an_dist2+self.margin_a
        diff2 = ap_dist2-nn_dist2+self.margin_b
        return torch.max(diff1,0).values\
               +torch.max(diff2,0).values


class CombinedTripletLosses(nn.Module):
    @staticmethod
    def add_criterion_specific_args(parent_parser):
        """Triplet Loss parameters
        Args:
            margin: margin of triplet loss
        """
        parser = parent_parser.add_argument_group("combined triplet loss params")
        parser.add_argument("--training.kkt_weights", type=list, default=[1.0, 0.0])
        return parent_parser
    def __init__(self, losses: list = [], kkt_weights: list = [0.0, 1.0]):
        super().__init__()
        n_losses = len(losses)
        n_weights = len(kkt_weights)
        assert n_losses == n_weights, f"number of losses ({n_losses} does NOT equal number of weights ({n_weights}))"
        self.losses = losses
        self.weights = torch.FloatTensor(kkt_weights) # torch.Size([n_weights])
    def forward(self, anchor, positive, negative) -> torch.float:
        # anchor.shape == positive.shape == negative.shape == [batch_size, feat_embedding]
        weights = self.weights.type_as(anchor)
        loss_outs = torch.concat([loss(anchor,positive,negative).view(1) for loss in self.losses]) # shape: torch.Size([n_weights])
        out = torch.dot(loss_outs, weights)
        return out


def args_per_criterion(parent_parser):
    losses = [TripletLoss, QuadrupletLoss, CombinedTripletLosses]
    for loss in losses:
        parent_parser = loss.add_criterion_specific_args(parent_parser)
    return parent_parser