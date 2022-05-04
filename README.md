## Introduction
This repository contains code to train Deep SORT's re-identification model with different optimization schemes. More spcifically, the optimization functions we use are:
* Triplet cosine using cosine and euclidean distance
* Weighted combination of triplet with cosine and euclidean
* Quadruplet loss

The design of this project is modular to allow the user to customize the configuration file to try different options. 

## Dependencies
The code has been tested only python 3.6. To install the required libraries, run below command:
```sh
pip install -r requirements.txt
```
> NOTE: we recommend creating a conda virtual environment to run this code `conda create -n deep_sort python=3.6`. 

## Datasets
For training and testing, we used the [Market-1501](http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) dataset which you can download by runing below:
```sh
wget -c http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip
```

For validation, we downloaded the training segment of the [MOT17 Dataset](https://motchallenge.net/data/MOT17/).

Once you have downloaded all the datasets, you should have a directory path like below:
```sh
Datasets
└─ Market-1501-v15.09.15
	├── bounding_box_test
	├── bounding_box_train
	├── . . .
    ├── query
└─ MOT17
    ├── test
		├── . . .
    ├── train
		├── . . .
```

## Pre-Processing
Since this work focuses on training the re-identification embedding of Deep SORT (and ignores the tracking framework), we format all our datasets into below format to easily mine triplet and quadruplet samples:
```sh
dataset
└─ 0001
	├── 0001_c1s1_001051_00.jpg
	├── 0001_c1s1_002301_00.jpg
	├── . . .
	├── 0001_c6s3_077467_00.jpg
└─ 0002
└─ . . .
```

Neither Market-1501 nor MOT17 come in the above dataset format. Further, while the images in the Market-1501 dataset are cropped images of the localization of a person, MOT17 comes in a video format (and thus, we need to crop the bounding box localizations in the video frames). To do this, run below commands
```sh
# turn Market-1501 to above format
python crop_dimensions.py --data_path /path/to/market1501/dataset --output_path out/path/ 


# turn MOT17 to above format
python cropper.py --mot-path /path/to/mot/training/dataset/partition/
```
> NOTE: for more help, you can always run something like `python cropper.py --help`. Further, for the output of `crop_dimensions.py`, you will have to manualy delete a `Thumbs` directory that includes a *.db file.


## Training Re-ID Embedding
To run training with the default configurations just run below:
```sh
python siamese_net.py --training.training_dir /market1501/training/path --validation.validation_dir /market1501/testing/path --testing.mot_testing_dir /mot17/train
```

If you want to customize the optimization function or any of the `Trainer` parameters from pytorch-lightning, you could do so by specifying the arguments through the command line but we highly recommend to create a configuration file like below:

`config.yaml`
```yaml
# enviornment parameters
env:
  exp_name: deepsort-triplet_cos
# training parameters
training:
  training_dir: /home/cougarnet.uh.edu/eeplater/Documents/Datasets/Market-1501-v15.09.15/bounding_box_train_pt_format
  criterion: triplet_cos
  batch_size: 128
  lr: .05
# validation parameters
validation:
  validation_dir: /home/cougarnet.uh.edu/eeplater/Documents/Datasets/Market-1501-v15.09.15/gt_bbox_pt_format
  batch_size: 256
# testing parameters
testing:
  mot_testing_dir: /home/cougarnet.uh.edu/eeplater/Documents/Datasets/MOT17/train
  batch_size: 256
# re-id model parameters
model:
  use_dropout: False
  act: elu
  blur: True
  arch_version: v0
# pl `Trainer` parameters
trainer:
  max_epochs: 1
  default_root_dir: /home/cougarnet.uh.edu/eeplater/Documents/GitHub/improved_Deep_SORT/data/models/trial1
  gpus: -1
  log_every_n_steps: 100
  profiler: simple
```

## References
* Deep SORT publication: https://arxiv.org/pdf/1703.07402.pdf

## Acknowledgements
We originally forked this repository from [here])(https://github.com/abhyantrika/nanonets_object_tracking). As such, a large part of this code such as `siamese_dataloader.py` originates from there. 