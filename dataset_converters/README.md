# Dataset Converters

Directory contains files that were used to convert datasets to other formats.



#### `mot_to_id.py`

File converts a MOT formatted dataset:

```bash
└─ MOT16/
    ├── train
    │   ├── MOT16-02/
    │   │   ├── det 
    │   │   │   ├── det.txt
    │   │   └── gt 
    │   │   │   ├── gt.txt
    │   │   ├── img1
    │   │   │   ├── 000001.jpeg
    │   │   │   ├── 000002.jpeg
    │   │   │   ├── . . .
    │   ├── MOT16-04/
    │       ├── . . .
    ├── test
    │   ├── ...
```

 to identity format:

```bash
└─ MOT16/
    ├── ID0
    │   ├── img0000.jpg
    │   ├── img0001.jpg
    │   ├── img0002.jpg
    ├── ID1
    │   ├── img0000.jpg
    │   ├── img0001.jpg
    │   ├── img0002.jpg
    ├── . . .
```

by running below:

```bash
python cropper.py MOT16/
```

Cropped images, in a folder for each identity, will be saved in a subfolder named "crops" in the MOT_VIDEO_DIRECTORY.



#### `market1501_to_id.py`

File converts Market-1501 format:

```bash
└─ bounding_box_train/
    ├── 0002_c1s1_000451_03.jpg
    ├── 0002_c1s1_000551_01.jpg
    ├── 0002_c1s1_000776_01.jpg
    ├── . . .
    ├── 0007_c2s3_070952_01.jpg
    ├── 0007_c2s3_071002_01.jpg
    ├── 0007_c2s3_071052_01.jpg
    ├── . . .
```

to identity format:

```bash
└─ MOT16/
    ├── ID0
    │   ├── img0000.jpg
    │   ├── img0001.jpg
    │   ├── img0002.jpg
    ├── ID1
    │   ├── img0000.jpg
    │   ├── img0001.jpg
    │   ├── img0002.jpg
    ├── . . .
```

#### `market1501_to_reid.py`

File converts Market-1501 format:

```bash
└─ bounding_box_train/
    ├── 0002_c1s1_000451_03.jpg
    ├── 0002_c1s1_000551_01.jpg
    ├── 0002_c1s1_000776_01.jpg
    ├── . . .
    ├── 0007_c2s3_070952_01.jpg
    ├── 0007_c2s3_071002_01.jpg
    ├── 0007_c2s3_071052_01.jpg
    ├── . . .
└─ gt_bbox
    ├── 0001_c1s1_001051_00.jpg
    ├── 0001_c1s1_002301_00.jpg
    ├── 0002_c1s1_002401_00.jpg
    ├── . . .
    ├── 0002_c1s1_000451_00.jpg
    ├── 0002_c1s1_000551_00.jpg
    ├── 0002_c1s1_000776_00.jpg
    ├── . . .
```

to re-identity format:

```bash
└─ MOT16/
    ├── ID0
    │   ├── img0000.jpg
    │   ├── img0001.jpg
    │   ├── img0002.jpg
    ├── ID1
    │   ├── img0000.jpg
    │   ├── img0001.jpg
    │   ├── img0002.jpg
    ├── . . .
└─ meta
    ├── train_80.txt
    ├── val_20.txt
```









