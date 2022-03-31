## Usage
```sh
python cropper.py MOT_VIDEO_DIRECTORY
```

The MOT_VIDEO_DIRECTORY should have the file structure:
```
train
├── MOT16-02
│   ├── det
│   ├── gt
│   ├── img1
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
│   │   ...
│   └── seqinfo.ini
├── MOT16-04
│   ├── det
│   ├── gt
│   ├── img1
│   └── seqinfo.ini
├── MOT16-05
│   ├── det
│   ├── gt
│   ├── img1
│   └── seqinfo.ini
│   ...
```

Cropped images, in a folder for each identity, will be saved in a subfolder named "crops" in the MOT_VIDEO_DIRECTORY.