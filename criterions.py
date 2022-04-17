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

    def forward(self, anchor, positive, negative, size_average=True):
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
    
    def __init__(self, margin_alpha=.1, margin_beta=.01, **kwargs):
        super(QuadrupletLoss, self).__init__()
        self.margin_a = margin_alpha
        self.margin_b = margin_beta

    def forward(self, ap_dist, an_dist, nn_dist):
        ap_dist2 = torch.square(ap_dist)
        an_dist2 = torch.square(an_dist)
        nn_dist2 = torch.square(nn_dist)
        return torch.sum(torch.max(ap_dist2-an_dist2+self.margin_a,dim=0),dim=0)\
               +torch.sum(torch.max(ap_dist2-nn_dist2+self.margin_b, dim=0),dim=0)


class CombinedTripletLosses(nn.Module):
    def __init__(self, *args):
        super().__init__()
        n_losses = len(args)
        self.losses = args
        self.weights = nn.Parameter(torch.FloatTensor(n_losses).uniform_(0,1))
    def forward(self, anchor, positive, negative):
        loss_outs = torch.FloatTensor([loss(anchor,positive,negative) for loss in self.losses])
        out = torch.dot(loss_outs, self.weights)
        return out


def args_per_criterion(parent_parser):
    losses = [TripletLoss, QuadrupletLoss]
    for loss in losses:
        parent_parser = loss.add_criterion_specific_args(parent_parser)
    return parent_parser