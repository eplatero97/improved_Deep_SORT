import torch
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
        parser.add_argument("--metrics.tripletacc.dist_thresh", type=float, default=0.2)
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
        # anchor.shape == positive.shape == negative.shape == (batch_size, feats)
        batch_size = anchor.shape[0]

        a2p_dist = self.dist(anchor, positive)
        a2n_dist = self.dist(anchor, negative)
        p2n_dist = self.dist(positive, negative)

        acc = 0
        acc += (a2p_dist <= self.dist_thresh).sum()
        acc += (a2n_dist > self.dist_thresh).sum()
        acc += (p2n_dist > self.dist_thresh).sum()
        acc_pct = acc / (3*batch_size) # mean accuracy percentage
        return acc_pct


class QuadrupletAcc:
    @staticmethod
    def add_metric_specific_args(parent_parser):
        """Triplet Loss parameters
        Args:
            margin: margin of triplet loss
        """
        parser = parent_parser.add_argument_group("triplet accuracy parameters")
        parser.add_argument("--metrics.quadrupletacc.p", type=float, default=2.0)
        parser.add_argument("--metrics.quadrupletacc.dist_thresh", type=float, default=0.2)
        return parent_parser
    
    def __init__(self, p: float=2.0, dist_thresh: float = .2):
        """
        :obj: calculate whether distance of pairs (a,p),(a,n),(p,n) are below `dist_thresh`
        :param p: norm degree
        :param dist_thresh: determines whether distance is enough to classify as same or different class of person
        """
        self.dist_thresh = dist_thresh
        self.dist = lambda x,y: LA.vector_norm(x-y, ord=p, dim=1)
        self.triplet = TripletAcc(p, dist_thresh)
    
    @torch.no_grad()
    def __call__(self, anchor: torch.FloatTensor, 
                       positive: torch.FloatTensor, 
                       negative: torch.FloatTensor,
                       negative2: torch.FloatTensor) -> float:
        
        batch_size = anchor.shape[0]
        a2n2_dist = self.dist(anchor, negative2)
        p2n2_dist = self.dist(positive, negative2)
        n2n2_dist = self.dist(negative, negative2)

        acc = 0
        tripletacc = self.triplet(anchor, positive, negative)
        acc += tripletacc / 3 # div by 3 to account for other 3 combinations of quadruplet loss
        acc += (a2n2_dist > self.dist_thresh).sum()
        acc += (p2n2_dist > self.dist_thresh).sum()
        acc += (n2n2_dist > self.dist_thresh).sum()
        acc_pct = acc / (6*batch_size) # accuracy percentage
        return acc_pct 


if __name__ == "__main__":
    a = torch.randn(3, 256)
    p = torch.randn(3, 256)
    n = torch.randn(3, 256)
    n2 = torch.randn(3, 256)

    acc = QuadrupletAcc(p=2.0, dist_thresh=0.2)
    out = acc(a,p,n,n2)
    # print(out)