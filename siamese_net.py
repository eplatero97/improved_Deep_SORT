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
import pytorch_lightning as pl
from typing import Optional
from siamese_dataloader import SiameseTriplet

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


class QuadrupletLoss(nn.Module):

    def __init__(self, margin_alpha, margin_beta, **kwargs):
        super(QuadrupletLoss, self).__init__()
        self.margin_a = margin_alpha
        self.margin_b = margin_beta

    def forward(self, ap_dist, an_dist, nn_dist):
        ap_dist2 = torch.square(ap_dist)
        an_dist2 = torch.square(an_dist)
        nn_dist2 = torch.square(nn_dist)
        return torch.sum(torch.max(ap_dist2-an_dist2+self.margin_a,dim=0),dim=0)\
               +torch.sum(torch.max(ap_dist2-nn_dist2+self.margin_b, dim=0),dim=0)


criterion = QuadrupletLoss(margin_alpha=0.1, margin_beta=0.01)

#Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask().cuda()

class MetricNetwork(LightningModule):
    def apply_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m)
    def __init__(self, single_embedding_shape):
        super(MetricNetwork, self).__init__()
        self.input_shape = single_embedding_shape
        self.input_shape[0]=self.input_shape*2
        ops = nn.ModuleList()
        ops.append(nn.Linear(self.input_shape,10))
        ops.append(nn.ReLU)
        ops.append(nn.Linear(10,10))
        ops.append(nn.ReLU)
        ops.append(nn.Linear(10,10))
        ops.append(nn.ReLU)
        ops.append(nn.Linear(10,2))
        ops.append(nn.Softmax)
        self.net = nn.Sequential(*ops)
        self.net.apply(self.apply_weights)

    def forward(self, x):
        out = self.net(x)
        return out[:, 0]





class SiameseNetwork(LightningModule):
    def __init__(self, use_dropout: bool = False, act = nn.ReLU):
        """Deep SORT encoder (siamese network)

        Args:
            use_dropout: whether to use dropout during training 
        """
        super(SiameseNetwork, self).__init__()
        # self.save_hyperparameters()

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
        self.metric_network = MetricNetwork(1024)
    def forward_once(self, x):
        output = self.net(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc(output)
        
        output = torch.squeeze(output)
        return output

    def forward(self, input1, input2,input3,input4=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        if input4 is not None:
            output4 = self.forward_once(input4)
            return output1,output2,output3, output4

        return output1, output2

    def training_step(self, batch, batch_idx):
		
        # Get anchor, positive and negative samples
        anchor, positive, negative, negative2 = batch
        anchor, positive, negative, negative2 = anchor.cuda(), positive.cuda() , negative.cuda(), negative2.cuda()
 
 		# Multiple image patches with gaussian mask. It will act as an attention mechanism which will focus on the center of the patch
        anchor, positive, negative, negative2 = anchor*gaussian_mask, positive*gaussian_mask, negative*gaussian_mask, negative2*gaussian_mask

        anchor_out, positive_out, negative_out, negative2_out = self(anchor, positive, negative, negative2) # Model forward propagation
        anchor_positive_out = torch.cat([anchor_out,positive_out],dim=-1)
        anchor_negative_out = torch.cat([anchor_out,negative_out],dim=-1)
        negative_negative2_out = torch.cat([negative_out,negative2_out],dim=-1)
        ap_dist = self.metric_network(anchor_positive_out)
        an_dist = self.metric_network(anchor_negative_out)
        nn_dist = self.metric_network(negative_negative2_out)


        quadruplet_loss: torch.float32 = criterion(ap_dist, an_dist, nn_dist) # Compute triplet loss (based on cosine simality) on the output feature maps
        self.log("train/triplet_loss",  quadruplet_loss.item(), logger = True, on_step = True, on_epoch = False)
        return quadruplet_loss


    def configure_optimizers(self):
        optim.Adam(self.parameters(), lr = 0.0005)



# define Deep SORT dataloader
class DeepSORTModule(pl.LightningDataModule):
	def __init__(self, data_path: str = "path/to/dir", batch_size: int = 32):
		"""Deep SORT Data Module

        Args:
            data_path: path to training/testing directory
            batch_size: batch size
        """
		super().__init__()
		self.root = data_path
		self.batch_size = batch_size
		self.transforms = transforms.Compose([
										transforms.Resize((256,128)),
										transforms.ColorJitter(hue=.05, saturation=.05),
										transforms.RandomHorizontalFlip(),
										transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
										transforms.ToTensor()
										])

	def setup(self, stage: Optional[str] = None):
		folder_dataset = dset.ImageFolder(root=self.root)
		self.siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset,
											  transform=self.transforms,should_invert=False) # Get dataparser class object
		

	def train_dataloader(self):
		return DataLoader(self.siamese_dataset,shuffle=True, 
						  num_workers=4,batch_size=self.batch_size) # PyTorch data parser obeject

