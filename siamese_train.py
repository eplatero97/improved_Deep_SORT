import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL.ImageOps    
from siamese_dataloader import *
from siamese_net import *
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Optional
from jsonargparse import ArgumentParser, ActionConfigFile
from pytorch_lightning.profiler.advanced import AdvancedProfiler
from torch import nn

seed_everything(42, workers=True)

"""
Get training data
"""

parser = ArgumentParser()

parser.add_argument("--root_dir", type = str, default = "lightning_logs", help = "root path to store logs")
parser.add_argument("--training_dir", type = str, default = "/media/ADAS1/MARS/bbox_train/bbox_train/")
parser.add_argument("--testing_dir", type = str, default = "/media/ADAS1/MARS/bbox_test/bbox_test/")
parser.add_argument("--train_batch_size", type = int, default = 128)
parser.add_argument("--use_dropout", type = bool, default = False)
parser.add_argument("--act", type=str, default = "relu", help = "activation layer to use for training encoder")
parser.add_argument("--deterministic", type=bool, default = True, help = "whether to make enable cudnn.deterministic or not. If True, will make training slower but it will better ensure reprodusability")
parser.add_argument("--config", action = ActionConfigFile) 

cfg = parser.parse_args()



# define Deep SORT dataloader
class DeepSORTModule(pl.LightningDataModule):
	def __init__(self, root: str = "path/to/dir", batch_size: int = 32):
		"""Deep SORT Data Module

        Args:
            root: path to training/testing directory
            batch_size: batch size
        """
		super().__init__()
		self.root = root
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
						  num_workers=14,batch_size=self.batch_size) # PyTorch data parser obeject



# init network and dataloader
if cfg.act == "relu":
	act = nn.ReLU

net = SiameseNetwork(use_dropout = cfg.use_dropout, act = act).cuda() # init model on gpu
train_datamodule = DeepSORTModule(cfg.training_dir, cfg.batch_size) # init training dataloader


# create model checkpoint every 20 epochs
checkpoint_callback = ModelCheckpoint(
    every_n_epochs = 20,
    dirpath = cfg.root_dir,
    filename="Deep-SORT-{epoch:02d}-{val_loss:.2f}"
)

# record time taken to execute each function with AdvancedProfiler

profiler = AdvancedProfiler(dirpath = cfg.root_dir, filename = "adv_profile.txt")

# init Trainer object (TensorBoard logger is used by default)
trainer = Trainer(default_root_dir = cfg.root_dir,
				  log_every_n_steps=10,
                  max_epochs = 100,
                  gpus = -1,
                  callbacks=[checkpoint_callback],
				  deterministic = cfg.deterministic,
				  profiler = profiler) 


# begin training
trainer.fit(net, train_datamodule)