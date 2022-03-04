import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from scipy.stats import multivariate_normal

def get_gaussian_mask():
	#128 is image size
	# We will be using 256x128 patch instead of original 128x128 path because we are using for pedestrain with 1:2 AR.
	x, y = np.mgrid[0:1.0:256j, 0:1.0:128j] #128 is input size.
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask


class ContrastiveLoss(torch.nn.Module):
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

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.cosine_similarity(anchor,positive) #Each is batch X 512 
        distance_negative = F.cosine_similarity(anchor,negative)  # .pow(.5)
        losses = (1- distance_positive)**2 + (0 - distance_negative)**2      #Margin not used in cosine case. 
        return losses.mean() if size_average else losses.sum()




criterion = TripletLoss(margin=1)

#Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask().cuda()


class SiameseNetwork(LightningModule):
    def __init__(self, use_dropout: bool = False, act = nn.ReLU):
        """Deep SORT encoder (siamese network)

        Args:
            use_dropout: whether to use dropout during training 
        """
        super(SiameseNetwork, self).__init__()

        #Outputs batch X 512 X 1 X 1 
        ops = nn.ModuleList()
        ops.append(nn.Conv2d(3,32,kernel_size=3,stride=2))
        ops.append(act())
        ops.append(nn.BatchNorm2d(32))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(nn.Conv2d(32,64,kernel_size=3,stride=2))
        ops.append(act())
        ops.append(nn.BatchNorm2d(64))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(nn.Conv2d(64,128,kernel_size=3,stride=2))
        ops.append(act())
        ops.append(nn.BatchNorm2d(128))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(nn.Conv2d(128,256,kernel_size=1,stride=2))
        ops.append(act())
        ops.append(nn.BatchNorm2d(256))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(nn.Conv2d(256,256,kernel_size=1,stride=2))
        ops.append(act())
        ops.append(nn.BatchNorm2d(256))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(nn.Conv2d(256,512,kernel_size=3,stride=2))
        ops.append(act())
        ops.append(nn.BatchNorm2d(512))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        
        ops.append(nn.Conv2d(512,1024,kernel_size=1,stride=1))
        ops.append(act())
        ops.append(nn.BatchNorm2d(1024))

        self.net = nn.Sequential(*ops)

    def forward_once(self, x):
        output = self.net(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc(output)
        
        output = torch.squeeze(output)
        return output

    def forward(self, input1, input2,input3=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if input3 is not None:
            output3 = self.forward_once(input3)
            return output1,output2,output3

        return output1, output2

    def training_step(self, batch, batch_idx):
		
        # Get anchor, positive and negative samples
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.cuda(), positive.cuda() , negative.cuda()
 
 		# Multiple image patches with gaussian mask. It will act as an attention mechanism which will focus on the center of the patch
        anchor, positive, negative = anchor*gaussian_mask, positive*gaussian_mask, negative*gaussian_mask

        anchor_out, positive_out, negative_out = self(anchor, positive, negative) # Model forward propagation

        triplet_loss = criterion(anchor_out, positive_out, negative_out) # Compute triplet loss (based on cosine simality) on the output feature maps
        self.log("performance", {"triplet_loss": triplet_loss.item()}, logger = True, on_step = True, on_epoch = False)
        return triplet_loss


    def configure_optimizers(self):
        optim.Adam(self.parameters(), lr = 0.0005)



