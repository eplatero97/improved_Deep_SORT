import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL.ImageOps    
from siamese_dataloader import *
from siamese_net import *
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

"""
Get training data
"""

class Config():
	training_dir = "/media/ADAS1/MARS/bbox_train/bbox_train/"
	testing_dir = "/media/ADAS1/MARS/bbox_test/bbox_test/"
	train_batch_size = 128


folder_dataset = dset.ImageFolder(root=Config.training_dir)

transforms = transforms.Compose([
	transforms.Resize((256,128)),
	transforms.ColorJitter(hue=.05, saturation=.05),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
	transforms.ToTensor()
	])






siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset,transform=transforms,should_invert=False) # Get dataparser class object
train_dataloader = DataLoader(siamese_dataset,shuffle=True,num_workers=14,batch_size=Config.train_batch_size) # PyTorch data parser obeject


net = SiameseNetwork().cuda() # Get model class object and put the model on GPU




# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    every_n_epochs = 20,
    dirpath = "lightning_logs/model_ckpts",
    filename="Deep-SORT-{epoch:02d}-{val_loss:.2f}"
)




trainer = Trainer(log_every_n_steps=10,
                  max_epochs = 100,
                  gpus = -1,
                  callbacks=[checkpoint_callback],
				  profiler = "advanced") 

trainer.fit(net, train_dataloader)