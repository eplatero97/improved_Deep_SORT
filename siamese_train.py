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
from criterions import * # imports all defined criterions
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
parser.add_argument("--data.training_dir", type = str, default = "/media/ADAS1/MARS/bbox_train/bbox_train/")
parser.add_argument("--data.testing_dir", type = str, default = "/media/ADAS1/MARS/bbox_test/bbox_test/")
parser = SiameseNetwork.add_model_specific_args(parser) # add model specific parameters
# add training specific configurations
parser.add_argument("--training.batch_size", type = int, default = 128)
parser.add_argument("--training.lr", type = float, default = 0.0005)
parser.add_argument("--training.criterion", type=str, default="triplet_cos")
parser = args_per_criterion(parser) # adds parameters of each defined criterion in `args_per_criterion`

cfg = Bunch(parser.parse_args()) # `parse_args()` returns dictionary of args
cfg.data = Bunch(cfg.data)
cfg.trainer = Bunch(cfg.trainer)
cfg.training = Bunch(cfg.training)
cfg.model = Bunch(cfg.model)

trainer_cfg = cfg.trainer
training_cfg = cfg.training
model_cfg = cfg.model


# define activation
model_act: str = model_cfg.act
if model_act == "elu":
	model_cfg.act = nn.ELU
elif model_act == "relu":
	model_cfg.act = nn.ReLU
elif model_act == "gelu":
	model_cfg.act = nn.GELU
else:
	print(f"your activation: {model_act} is not defined")
	raise 

# define criterion
criterion_name, criterion_metric = training_cfg.criterion.split('_')
if criterion_name == "triplet":
	if criterion_metric == "cos":
		training_cfg.criterion = TripletLoss() # cosine does NOT need margin
	elif criterion.metric == "eucl":
		margin = training_cfg.triplet.margin
		training_cfg.criterion = nn.TripletMarginLoss(margin=margin)
	else:
		print(f"Triplet loss with specified metric of {criterion_metric} is NOT defined")
		raise
	
elif criterion_name == "quad":
	if criterion_metric == "learnedmetric":
		margin_alpha = cfg.training.quadruplet.margin_alpha
		margin_beta = cfg.training.quadruplet.margin_beta
		cfg.training.criterion = QuadrupletLoss(margin_alpha=margin_alpha, margin_beta=margin_beta) 
	else:
		print(f"Triplet loss with specified metric of {criterion_metric} is NOT defined")
		raise

# print configurations
#print(f"Configurations: \n{vars(cfg)}")
#print(f"model configs: {vars(cfg.model)}")

net = SiameseNetwork(cfg).cuda() # init model on gpu
train_datamodule = DeepSORTModule(cfg.data.training_dir, cfg.training.batch_size, cfg.model.arch_version) # init training dataloader


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