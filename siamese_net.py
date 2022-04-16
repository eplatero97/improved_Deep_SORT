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



"""
using below blur is really slow to perform on cpu. 
As such, authors move it to the forward part of the model
to perform operation on gpu 
"""
class get_gaussian_mask:
    def __init__(self, dim0 = 256, dim1 = 128, cuda: bool = False):
        
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

        if cuda:
            # `img` is also expected to be cuda
            self.mask = torch.from_numpy(z).cuda()
        else:
            self.mask = torch.from_numpy(z)
    
    @torch.no_grad()
    def __call__(self, img):
        #Multiply each image with mask to give attention to center of the image.
        # Multiple image patches with gaussian mask. It will act as an attention mechanism which will focus on the center of the patch
        return self.mask * img 


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
        distance_positive = F.cosine_similarity(anchor,positive) # maximize 
        distance_negative = F.cosine_similarity(anchor,negative)  # minimize (can take to the power of `.pow(.5)`)
        losses = (1- distance_positive)**2 + (0 - distance_negative)**2      #Margin not used in cosine case. 
        return losses.mean() if size_average else losses.sum()




class BasicBlock(LightningModule):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1, act=nn.ReLU):
        super().__init__()
        conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        activation = act()
        bn = nn.BatchNorm2d(out_channels)
        self.block = nn.Sequential(conv, activation, bn)

    def forward(self, x):
        return self.block(x)

# https://peiyihung.github.io/mywebsite/category/learning/2021/08/22/build-resnet-from-scratch-with-pytorch.html
class ResidualBlock(nn.Module):
    def __init__(self, ni: int, nf: int):
        
        super().__init__()
        
        # shorcut
        if ni < nf: 
            
            # change channel size
            self.shortcut = nn.Sequential(
                nn.Conv2d(ni, nf, kernel_size=1, stride=2),
                nn.BatchNorm2d(nf))
            
            # downsize the feature map
            first_stride = 2
        else:
            self.shortcut = lambda x: x
            first_stride = 1
        
        # convnet
        self.conv = nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=3, stride=first_stride, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf)
        )
    
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class SiameseNetwork(LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """Deep SORT encoder (siamese network)

        Args:
            use_dropout: whether to use dropout during training 
            lr: learning rate
            criterion: select which loss function to use during training
            act: specify which activation function to use
            blur: whether to blur image before creating embedding to focus attention on center of content 
            arch_version: select which architecture to choose from
        """
        parser = parent_parser.add_argument_group("reid")
        parser.add_argument("--use_dropout", type = bool, default = False)
        parser.add_argument("--lr", type = float, default = 0.0005)
        parser.add_argument("--criterion", type = str, default = "triplet_cos")
        parser.add_argument("--act", type=str, default = "relu")
        parser.add_argument("--blur", type=bool, default=True)
        parser.add_argument("--arch_version", type=str, default="v1")
        return parent_parser
    
    def __init__(self, cfg):

        super(SiameseNetwork, self).__init__()
        # self.save_hyperparameters()
        self.use_dropout: bool = cfg.use_dropout
        self.lr: float = cfg.lr
        self.criterion: nn.modules.loss = cfg.criterion
        self.act: nn.modules.activation = cfg.act
        self.blur: bool = cfg.blur
        if self.blur:
            self.gaussian_mask = get_gaussian_mask(cuda = True)

        # initiate model
        if cfg.arch_version == "v1":
            self.init_archv1()
        elif cfg.arch_version == "v2":
            self.init_archv2()
        elif cfg.arch_version == "v3":
            self.init_archv3()

    def init_archv1(self):
        
        # define model parameters
        act: nn.modules.activation = self.act
        use_dropout: bool = self.use_dropout


        ops = nn.ModuleList()
        ops.append(BasicBlock(3,32,kernel_size=3,stride=2, act=act)) # shape: torch.Size([batch_size, 32, 127, 63])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(32,64,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 64, 63, 31])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(64,128,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 128, 31, 15])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(128,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 16, 8])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 8, 4])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,512,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 512, 3, 1])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        

        ops.append(BasicBlock(512,1024,kernel_size=(3,1),stride=1,act=act)) # shape: torch.Size([batch_size, 1024, 1, 1])

        self.net = nn.Sequential(*ops)

    def init_archv2(self):
        """
        version is identical to v1 but we replaced each basic block with a residual block for every operation that contains
        `kernel_size=1`
        """
        
        # define model parameters
        act: nn.modules.activation = self.act
        use_dropout: bool = self.use_dropout


        ops = nn.ModuleList()
        ops.append(BasicBlock(3,32,kernel_size=3,stride=2, act=act)) # shape: torch.Size([batch_size, 32, 127, 63])
        ops.append(ResidualBlock(32,32))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(32,64,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 64, 63, 31])
        ops.append(ResidualBlock(64,64))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(64,128,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 128, 31, 15])
        ops.append(ResidualBlock(128,128))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(128,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 16, 8])
        ops.append(ResidualBlock(256,256))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,256,kernel_size=1,stride=2,act=act)) # shape: torch.Size([batch_size, 256, 8, 4])
        ops.append(ResidualBlock(256,256))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,512,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 512, 3, 1])
        ops.append(ResidualBlock(512,512))
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        

        ops.append(BasicBlock(512,1024,kernel_size=(3,1),stride=1,act=act)) # shape: torch.Size([batch_size, 1024, 1, 1])

        self.net = nn.Sequential(*ops)


    def init_archv3(self):
        """
        version is identical to v1 but we replaced each basic block with a residual block for every operation that contains
        `kernel_size=1`
        """
        
        # define model parameters
        act: nn.modules.activation = self.act
        use_dropout: bool = self.use_dropout


        ops = nn.ModuleList()
        ops.append(BasicBlock(3,32,kernel_size=3,stride=2, act=act)) # shape: torch.Size([batch_size, 32, 127, 63])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(32,64,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 64, 63, 31])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(64,128,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 128, 31, 15])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(ResidualBlock(128,256))
        ops.append(nn.MaxPool2d(1,stride=2,padding=0)) # shape: torch.Size([batch_size, 256, 16, 8])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(ResidualBlock(256,256))
        ops.append(nn.MaxPool2d(1,stride=2,padding=0)) # shape: torch.Size([batch_size, 256, 8, 4])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        ops.append(BasicBlock(256,512,kernel_size=3,stride=2,act=act)) # shape: torch.Size([batch_size, 512, 3, 1])
        if use_dropout:
            ops.append(nn.Dropout2d(p=0.4))
        

        ops.append(BasicBlock(512,1024,kernel_size=(3,1),stride=1,act=act)) # shape: torch.Size([batch_size, 1024, 1, 1])

        self.net = nn.Sequential(*ops)

    def forward_once(self, x):
        # x.shape: torch.Size([batch_size,3,256,128])
        
        batch_size: int = x.shape[0]
        output = self.net(x) # shape: torch.Size([batch_size,1024,1,1]) 
        #output = output.view(output.size()[0], -1)
        #output = self.fc(output)
        output = torch.squeeze(output) # shape: torch.Size([batch_size, 1024])

        if batch_size == 1:
            # if True, then output shape will be: torch.Size([1024])
            #  thus, we add the batch dimension again to be torch.Size([1,1024])
            return output.view(1,-1)
        
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

        if self.blur:
            anchor, positive, negative = self.gaussian_mask(anchor), self.gaussian_mask(positive), self.gaussian_mask(negative)

        anchor_out, positive_out, negative_out = self(anchor, positive, negative) # Model forward propagation

        triplet_loss: torch.float32 = self.criterion(anchor_out, positive_out, negative_out) # Compute triplet loss (based on cosine simality) on the output feature maps
        self.log("train/triplet_loss",  triplet_loss.item(), logger = True, on_step = True, on_epoch = False)
        return triplet_loss


    def configure_optimizers(self):
        optim.Adam(self.parameters(), lr = self.lr)




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
                                        # get_gaussian_mask()
										])

	def setup(self, stage: Optional[str] = None):
		folder_dataset = dset.ImageFolder(root=self.root)
		self.siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset,
											  transform=self.transforms,should_invert=False) # Get dataparser class object
		

	def train_dataloader(self):
		return DataLoader(self.siamese_dataset,shuffle=True, 
						  num_workers=4,batch_size=self.batch_size) # PyTorch data parser obeject

