import torch
from torch import nn
#from torch.linalg import vector_norm
from torch.linalg import norm

a = torch.ones(1,100)
p = torch.zeros(1,100)
n = torch.ones(1,100)



triplet = nn.TripletMarginLoss(margin=0.0)

#ap_dist = vector_norm(a-p,dim=1)
#an_dist = vector_norm(a-n,dim=1)
ap_dist = norm(a-p,dim=1)
an_dist = norm(a-n,dim=1)

sim_triplet = ap_dist - an_dist
print(f"sim_triplet: {sim_triplet}")

triplet_loss = triplet(a,p,n)
print(f"triplet: {triplet_loss}")