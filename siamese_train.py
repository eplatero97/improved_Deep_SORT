#import torchvision.datasets as dset
from torchvision import transforms as T 
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
from callbacks import * # MyPrintingCallback
import wandb
from loguru import logger 
from utils import * # wrap_namespace (turns nested dictionary into namespace)
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

# extract dictionary arguments
args = parser.parse_args() # type: dict

# load default parameters
cfg = YamlParser(config_file="./cfgs/default_cfg.yml")

# load defined parameters 
cfg.merge_from_dict(args)

# init wandb and pass configurations to wandb
wandb_logger = WandbLogger(project="smart-bus", name = "deepsort-nvidiaai", log_model = "all")
wandb_logger.experiment.config.update(cfg)

# turn nested dictionary into namespace
cfg = wrap_namespace(cfg)


# unpack nested Namespace cfgs
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
	
elif criterion_name == "quadruplet":
	if criterion_metric == "learnedmetric":
		margin_alpha = cfg.training.quadruplet.margin_alpha
		margin_beta = cfg.training.quadruplet.margin_beta	
		cfg.training.criterion = QuadrupletLoss(margin_alpha=margin_alpha, margin_beta=margin_beta) 
	else:
		print(f"Triplet loss with specified metric of {criterion_metric} is NOT defined")
		raise


# init model on gpu
net = SiameseNetwork(cfg).cuda() 

# define input transformations and then create dataloader
transforms = T.Compose([
		T.Resize((128,64)) if cfg.model.arch_version == "v0" else T.Resize((256,128)),
		T.ColorJitter(hue=.05, saturation=.05),
		T.RandomHorizontalFlip(),
		T.RandomRotation(20, resample=PIL.Image.BILINEAR),
		T.ToTensor()
		# get_gaussian_mask()
		])
train_datamodule = DeepSORTModule(cfg.data.training_dir, cfg.training.batch_size, transforms, criterion_name) # init training dataloader


# create model checkpoint every 20 epochs
checkpoint_callback = ModelCheckpoint(
    every_n_epochs = 1,
    dirpath = trainer_cfg.default_root_dir,
    filename="Deep-SORT-{epoch:02d}-{val_loss:.2f}"
)



# add all custom trainer configurations
trainer_cfg.callbacks = [checkpoint_callback, MyPrintingCallback()]
trainer_cfg.logger = wandb_logger

# init Trainer with configuration parameters
trainer = Trainer.from_argparse_args(trainer_cfg)

# begin training
trainer.fit(net, train_datamodule)