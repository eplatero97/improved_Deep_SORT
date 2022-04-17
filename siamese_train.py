#import torchvision.datasets as dset
#from torchvision import transforms
from torch.utils.data import DataLoader
import PIL.ImageOps    
from siamese_dataloader import *
from siamese_net import *
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from typing import Optional
from pytorch_lightning.profiler.advanced import AdvancedProfiler
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cli import LightningArgumentParser
from torch import nn
import wandb
from loguru import logger 
seed_everything(42, workers=True)

# turn dictionary into namespace object
class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

"""
Get training data
"""

parser = LightningArgumentParser()

parser.add_lightning_class_args(Trainer, "trainer") # add trainer args
parser.add_argument("--training_dir", type = str, default = "/media/ADAS1/MARS/bbox_train/bbox_train/")
parser.add_argument("--testing_dir", type = str, default = "/media/ADAS1/MARS/bbox_test/bbox_test/")
parser.add_argument("--train_batch_size", type = int, default = 128)
parser = SiameseNetwork.add_model_specific_args(parser) # add model specific parameters

cfg = Bunch(parser.parse_args()) # `parse_args()` returns dictionary of args
trainer_cfg = Bunch(cfg.trainer)


# define activation
if cfg.act == "relu":
	cfg.act = nn.ReLU
elif cfg.act == "gelu":
	cfg.act = nn.GELU
else:
	print(f"your activation: {cfg.act} is not defined")
	raise 

# define criterion
if cfg.criterion == "triplet_cos":
	cfg.criterion = TripletLoss(margin=1)
elif cfg.criterion == "triplet_eucl":
	cfg.criterion = nn.TripletMarginLoss()
elif cfg.criterion == "quad_metric":
	cfg.criterion = QuadrupletLoss(margin_alpha=0.1, margin_beta=0.01) 

net = SiameseNetwork(cfg).cuda() # init model on gpu
train_datamodule = DeepSORTModule(cfg.training_dir, cfg.train_batch_size) # init training dataloader


# create model checkpoint every 20 epochs
checkpoint_callback = ModelCheckpoint(
    every_n_epochs = 20,
    dirpath = trainer_cfg.default_root_dir,
    filename="Deep-SORT-{epoch:02d}-{val_loss:.2f}"
)


# create custom calls during training
class MyPrintingCallback(Callback):  
  
	def on_init_start(self, trainer):  
		print("initializing `Trainer()` object")  

	def on_init_end(self, trainer):  
		print('`Trainer()` has been initialized')
		print("Begin Training")  

	def on_train_end(self, trainer, pl_module):  
		print('Training has Finished') 



# init wandb and pass configurations to wandb
wandb_logger = WandbLogger(project="smart-bus", name = "deepsort-nvidiaai", log_model = "all")
wandb_logger.experiment.config.update(cfg)

# add all custom trainer configurations
trainer_cfg.callbacks = [checkpoint_callback, MyPrintingCallback()]
trainer_cfg.profiler = "simple"
trainer_cfg.logger = wandb_logger

# init Trainer with configuration parameters
trainer = Trainer.from_argparse_args(trainer_cfg)

# begin training
trainer.fit(net, train_datamodule)