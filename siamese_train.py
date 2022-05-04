#import torchvision.datasets as dset
from pathlib import Path
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
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.cli import LightningArgumentParser
from torch import nn
from criterions import * # imports all defined criterions
from callbacks import * # MyPrintingCallback
import wandb
from loguru import logger 
from utils import * # wrap_namespace (turns nested dictionary into namespace)
import os
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
parser.add_argument("--env.exp_name", type = str, default = "experiment_name", help="experiment name")
parser.add_argument("--training.training_dir", type = str, default = "/media/ADAS1/MARS/bbox_train/bbox_train/")
parser.add_argument("--validation.validation_dir", type = str, default = "/media/ADAS1/MARS/bbox_test/bbox_test/")
parser.add_argument("--validation.batch_size", type = int, default = 256)
parser.add_argument("--testing.mot_testing_dir", type = str, default = "/media/train_mot/")
parser.add_argument("--testing.batch_size", type = int, default = 256)
parser = SiameseNetwork.add_model_specific_args(parser) # add model specific parameters
# add training specific configurations
parser.add_argument("--training.dataset", type=str, default="market1501")
parser.add_argument("--training.batch_size", type = int, default = 128)
parser.add_argument("--training.lr", type = float, default = 0.0005)
parser.add_argument("--training.criterion", type=str, default="triplet_cos")
parser = args_per_criterion(parser) # adds parameters of each defined criterion in `args_per_criterion`
parser = args_per_metric(parser) # adds parameters of each metric defined in `metrics.py`
parser.add_argument("--validation.dataset", type=str, default="mot17half")
parser.add_argument("--testing.dataset", type=str, default="mot17half")

# extract dictionary arguments
args = parser.parse_args() # type: dict

# load default parameters
cfg = YamlParser(config_file="./cfgs/default_cfg.yml")

# load defined parameters 
cfg.merge_from_dict(args)

# init wandb and pass configurations to wandb
wandb_logger = WandbLogger(project="smart-bus", name = cfg["env"]["exp_name"], log_model = True)
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
	elif criterion_metric == "eucl":
		margin = training_cfg.triplet.margin
		training_cfg.criterion = nn.TripletMarginLoss(margin=margin)
	elif criterion_metric == "comb":
		margin = training_cfg.triplet.margin
		loss0 = TripletLoss()
		loss1 = nn.TripletMarginLoss(margin=margin)
		losses = [loss0, loss1]
		weights = training_cfg.kkt_weights
		training_cfg.criterion = CombinedTripletLosses(losses, weights)

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

# define input transformations 
transforms = T.Compose([
		T.Resize((128,64)) if cfg.model.arch_version == "v0" else T.Resize((256,128)),
		T.ColorJitter(hue=.05, saturation=.05),
		T.RandomHorizontalFlip(),
		T.RandomRotation(20, resample=PIL.Image.BILINEAR),
		T.ToTensor()
		# get_gaussian_mask()
		])

# define train/validation/test dataloaders thorugh lightning datamodule
training_path=cfg.training.training_dir
validation_path=cfg.validation.validation_dir
testing_path= list(Path(cfg.testing.mot_testing_dir).rglob("crops_gt"))
training_batch_size=cfg.training.batch_size
validation_batch_size=cfg.validation.batch_size
testing_batch_size=cfg.testing.batch_size
datamodule = DeepSORTModule(training_path=training_path, 
								  validation_path=validation_path,
								  testing_path=testing_path,
								  training_batch_size=training_batch_size,
								  validation_batch_size=validation_batch_size,
								  testing_batch_size=testing_batch_size,
								  transforms=transforms,
								  mining=criterion_name)


# create model checkpoint every epoch
checkpoint_callback = ModelCheckpoint(
    every_n_epochs = 1,
    dirpath = os.path.join(trainer_cfg.default_root_dir, "ckpts"),
    filename="Deep-SORT-{epoch:02d}-{val_loss:.2f}"
)



# add all custom trainer configurations
trainer_cfg.callbacks = [checkpoint_callback, MyPrintingCallback()]
trainer_cfg.logger = wandb_logger

# init Trainer with configuration parameters
trainer = Trainer.from_argparse_args(trainer_cfg)

# begin training
trainer.fit(net, datamodule=datamodule)
trainer.test(datamodule=datamodule, ckpt_path="best")
# trainer.test(net, datamodule=datamodule)