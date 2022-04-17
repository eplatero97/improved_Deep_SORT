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
from reid_architectures import *


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





# criterion = QuadrupletLoss(margin_alpha=0.1, margin_beta=0.01) criterion is now defined in `siamese_train.py`




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





class SiameseNetwork(ReID_Architectures):
    
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
        parser.add_argument("--model.use_dropout", type = bool, default = False)
        parser.add_argument("--model.act", type=str, default = "relu")
        parser.add_argument("--model.blur", type=bool, default=True)
        parser.add_argument("--model.arch_version", type=str, default="v0")
        return parent_parser
    
    def __init__(self, cfg):

        super(SiameseNetwork, self).__init__()
        # self.save_hyperparameters()
        self.use_dropout: bool = cfg.model.use_dropout
        self.lr: float = cfg.training.lr
        self.criterion: nn.modules.loss = cfg.training.criterion
        self.criterion_name: str = type(cfg.criterion).__name__
        self.act: nn.modules.activation = cfg.model.act
        self.blur: bool = cfg.model.blur
        if self.blur:
            self.gaussian_mask = get_gaussian_mask(cuda = True)

        # define metric network
        if self.criterion_name == "QuadrupletLoss":
            self.metric_network = MetricNetwork(1024)

        # initiate model
        if cfg.arch_version == "v0":
            self.init_archv0()
        elif cfg.arch_version == "v1":
            self.init_archv1()
        else:
            print(f"Specified version is NOT defined")
            raise 


        
    def forward_once(self, x):
        # x.shape: Union[torch.Size([batch_size,3,256,128]), torch.Size([batch_size,3,128,64])]
        return self.net(x) # shape: torch.Size([batch_size, feat_dim]) 

    def forward(self, input1, input2,input3,input4=None):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        if input4 is not None:
            output4 = self.forward_once(input4)
            return output1,output2,output3, output4

        return output1,output2,output3

    def training_step(self, batch, batch_idx):

        if self.criterion_name == "TripletLoss":
            # Get anchor, positive and negative samples
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.cuda(), positive.cuda() , negative.cuda()

            if self.blur:
                anchor, positive, negative = self.gaussian_mask(anchor), self.gaussian_mask(positive), self.gaussian_mask(negative)
            
            anchor_out, positive_out, negative_out = self(anchor, positive, negative)
            triplet_loss: torch.float32 = self.criterion(anchor_out, positive_out, negative_out) # Compute triplet loss (based on cosine simality) on the output feature maps
            self.log(f"train/{self.criterion_name}",  triplet_loss.item(), logger = True, on_step = True, on_epoch = False)
            return triplet_loss
		elif self.criterion_name == "QuadrupletLoss":
            # get anchor, positive, negative and negative2 embeddings
            anchor, positive, negative, negative2 = batch
            anchor, positive, negative, negative2 = anchor.cuda(), positive.cuda() , negative.cuda(), negative2.cuda()
    
            # Multiple image patches with gaussian mask. It will act as an attention mechanism which will focus on the center of the patch
            if self.blur:
                anchor, positive, negative, negative2 = anchor*gaussian_mask, positive*gaussian_mask, negative*gaussian_mask, negative2*gaussian_mask

            anchor_out, positive_out, negative_out, negative2_out = self(anchor, positive, negative, negative2) # Model forward propagation
            anchor_positive_out = torch.cat([anchor_out,positive_out],dim=-1)
            anchor_negative_out = torch.cat([anchor_out,negative_out],dim=-1)
            negative_negative2_out = torch.cat([negative_out,negative2_out],dim=-1)
            ap_dist = self.metric_network(anchor_positive_out)
            an_dist = self.metric_network(anchor_negative_out)
            nn_dist = self.metric_network(negative_negative2_out)

            quadruplet_loss: torch.float32 = criterion(ap_dist, an_dist, nn_dist) # Compute triplet loss (based on cosine simality) on the output feature maps
            self.log(f"train/{self.criterion_name}",  quadruplet_loss.item(), logger = True, on_step = True, on_epoch = False)
            return quadruplet_loss

    def configure_optimizers(self):
        optim.Adam(self.parameters(), lr = self.lr)

    def configure_criterions(self):
        if type(self.criterion) == str:
            criterions = self.criterion.split('+')




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

