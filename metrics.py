import torch
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
        self.dist = lambda x,y: (x-y).pow(p).sum().pow(1/p)
    
    @torch.no_grad()
    def __call__(self, anchor: torch.FloatTensor, 
                       positive: torch.FloatTensor, 
                       negative: torch.FloatTensor) -> float:
        a2p_dist = self.dist(anchor, positive)
        a2n_dist = self.dist(anchor, negative)
        p2n_dist = self.dist(positive, negative)

        acc = 0
        acc += 1 if a2p_dist <= self.dist_thresh else 0
        acc += 1 if a2n_dist > self.dist_thresh else 0
        acc += 1 if p2n_dist > self.dist_thresh else 0
        acc_pct = acc / 3 # accuracy percentage
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
        self.dist = lambda x,y: (x-y).pow(p).sum().pow(1/p)
        self.triplet = TripletAcc(p, dist_thresh)
    
    @torch.no_grad()
    def __call__(self, anchor: torch.FloatTensor, 
                       positive: torch.FloatTensor, 
                       negative: torch.FloatTensor,
                       negative2: torch.FloatTensor) -> float:
        
        tripletacc = self.triplet(anchor, positive, negative)
        a2n2_dist = self.dist(anchor, negative2)
        p2n2_dist = self.dist(positive, negative2)
        n2n2_dist = self.dist(negative, negative2)

        acc = 0
        acc += 1 if a2n2_dist > self.dist_thresh else 0
        acc += 1 if p2n2_dist > self.dist_thresh else 0
        acc += 1 if n2n2_dist > self.dist_thresh else 0
        acc_pct = acc / 3 # accuracy percentage
        return acc_pct + tripletacc

