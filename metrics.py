import torch
from siamese_net import *
from torch import linalg as LA

# metric for Triplet Accuracy
class TripletAcc:
    @staticmethod
    def add_metric_specific_args(parent_parser):
        """Triplet Loss parameters
        Args:
            margin: margin of triplet loss
        """
        parser = parent_parser.add_argument_group("triplet accuracy parameters")
        parser.add_argument("--metrics.tripletacc.p", type=float, default=2.0)
        # parser.add_argument("--metrics.tripletacc.dist_thresh", type=float, default=0.2)
        return parent_parser
    
    def __init__(self, p: float=2.0, dist_thresh: float = .2):
        """
        :obj: calculate whether distance of pairs (a,p),(a,n),(p,n) are below `dist_thresh`
        :param p: norm degree
        :param dist_thresh: determines whether distance is enough to classify as same or different class of person
        """
        self.dist_thresh = dist_thresh
        self.dist = lambda x,y: LA.vector_norm(x-y, ord=p, dim=1)
    
    @torch.no_grad()
    def __call__(self, anchor: torch.FloatTensor, 
                       positive: torch.FloatTensor, 
                       negative: torch.FloatTensor) -> float:
        """
        :obj: calculate triplet accuracy on anchor,positive,negative embeddings
        """
        # anchor.shape == positive.shape == negative.shape == (batch_size, feats)
        batch_size = anchor.shape[0]

        a2p_dist = self.dist(anchor, positive) # shape: [batch_size,]
        a2n_dist = self.dist(anchor, negative) # shape: [batch_size,]
        p2n_dist = self.dist(positive, negative) # shape: [batch_size,]

        a_dists = torch.stack((a2p_dist, a2n_dist)) # shape: [2,batch_size]
        p_dists = torch.stack((a2p_dist, p2n_dist)) # shape: [2,batch_size]

        acc = 0.0
        acc += (a_dists.min(dim=0).indices == 0).sum()
        acc += (p_dists.min(dim=0).indices == 0).sum()
        acc /= (2*batch_size) # mean accuracy percentage
        return acc


class QuadrupletAcc:
    # @staticmethod
    # def add_metric_specific_args(parent_parser):
    #     """Quadruplet Accuracy parameters
    #     Args:
    #         margin_alpha: margin between pairs (a,p) and (a,n)
    #         margin_beta: margin between pairs (a,p) and (n,n2)
    #     """
    #     parser = parent_parser.add_argument_group("triplet accuracy parameters")
    #     parser.add_argument("--metrics.quadrupletacc.margin_alpha", type=float, default=0.2)
    #     parser.add_argument("--metrics.quadrupletacc.margin_beta", type=float, default=0.1)
    #     return parent_parser
    
    def __init__(self, learned_dist): # , margin_alpha=2.0, margin_beta=2.0):
        """
        :obj: calculate nearest neighbor
        :param learned_dist: learned distance function used to train qaudruplet loss
        :param dist_thresh: determines whether distance is enough to classify as same or different class of person
        """
        # self.margin_alpha = margin_alpha
        # self.margin_beta = margin_beta
        self.dist = learned_dist
    
    @torch.no_grad()
    def __call__(self, anchor: torch.FloatTensor, 
                       positive: torch.FloatTensor, 
                       negative: torch.FloatTensor,
                       negative2: torch.FloatTensor) -> float:
        """
        :obj: calculate quadruplet accuracy on anchor,positive,negative embeddings
        """
        # anchor.shape == positive.shape == negative.shape == negative2.shape == [batch_size, feats]
        batch_size = anchor.shape[0] 
        anchor_positive_out = torch.cat([anchor,positive],dim=-1) # shape: [batch_size, feats*2]
        anchor_negative_out = torch.cat([anchor,negative],dim=-1) # shape: [batch_size, feats*2]
        anchor_negative2_out = torch.cat([anchor,negative2],dim=-1) # shape: [batch_size, feats*2]
        positive_negative_out = torch.cat([positive,negative],dim=-1) # shape: [batch_size, feats*2]
        positive_negative2_out = torch.cat([positive,negative2],dim=-1) # shape: [batch_size, feats*2]
        negative_negative2_out = torch.cat([negative,negative2],dim=-1) # shape: [batch_size, feats*2]
        # feed to learned metric network
        # ap_dist.shape == ... == nn2_dist.shape == [batch_size]
        ap_dist = self.dist(anchor_positive_out).square() # shape: batch_size (same identity)
        an_dist = self.dist(anchor_negative_out).square() # shape: batch_size (different identity)
        an2_dist = self.dist(anchor_negative2_out).square() # shape: batch_size (different identity)
        pn_dist = self.dist(positive_negative_out).square() # shape: batch_size (different identity)
        pn2_dist = self.dist(positive_negative2_out).square() # shape: batch_size (different identity)
        # nn2_dist = self.dist(negative_negative2_out).square() # shape: batch_size (different identity)

        a_dists = torch.stack((ap_dist, an_dist, an2_dist)) # shape: [3,batch_size]
        p_dists = torch.stack([ap_dist, pn_dist, pn2_dist]) # shape: [3,batch_size]

        acc = 0.0
        acc += (a_dists.min(dim=0).indices == 0).sum()
        acc += (p_dists.min(dim=0).indices == 0).sum()
        acc /= (batch_size*2) # accuracy percentage
        return acc 

def args_per_metric(parent_parser):
    metrics = [TripletAcc] #, QuadrupletAcc]
    for metric in metrics:
        parent_parser = metric.add_metric_specific_args(parent_parser)
    return parent_parser

if __name__ == "__main__":
    learned_metric = MetricNetwork(128)
    acc = QuadrupletAcc(learned_metric, 2.0, 2.0)
    a = torch.randn(3,128)
    p = torch.randn(3,128)
    n = torch.randn(3,128)
    n2 = torch.randn(3,128)
    out = acc(a,p,n,n2)
    print(out)