from pytorch_lightning.callbacks import Callback

# create custom calls during training
class MyPrintingCallback(Callback):  
  
	def on_init_start(self, trainer):  
		# called once we are about to initialize `Trainer()` object
		print(f"Trainer() Configurations: {trainer}")  

	def on_init_end(self, trainer):  
		# called once `Trainer()` object is done initializing
		print('`Trainer()` has been initialized')
		print("Begin Training")  

	def on_train_start(self, trainer, pl_module):
		# called once we are to begin training 
		print(f"################ START TRAINING ################")

	def on_train_end(self, trainer, pl_module):  
		# called once training is over
		print(f"################ END TRAINING ################")