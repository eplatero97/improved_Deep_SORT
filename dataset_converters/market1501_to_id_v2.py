import os
import shutil
from jsonargparse import ArgumentParser, ActionConfigFile
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--market_path", type=str, required=True, help="Path to Market-1501 dataset")
    parser.add_argument("--is_train", type=bool, required=True, default=True, help="whether to convert training (True) or Test (False) dataset")
    parser.add_argument("--out_dir", type=str, required=False, help="Output path to store human identities", default=None)
    parser.add_argument("--config", action = ActionConfigFile)
    args = parser.parse_args()
    return args

def organize(src: str, dest: str):
    for file in os.listdir(src):
        identity = file.split("_")[0]
        identity_dir = os.path.join(dest, identity)
        if not os.path.exists(identity_dir):
            os.mkdir(identity_dir)
        if file not in os.listdir(identity_dir):
            file_path = os.path.join(src, file)
            shutil.copy(file_path, identity_dir)

def main():
    args = parse_args()

    if args.is_train:
        bbox_dir = os.path.join(args.market_path, "bounding_box_train")
        phase = "train"
    else:
        bbox_dir = os.path.join(args.market_path, "bounding_box_test")
        phase = "test"

    if args.out_dir == None:
        pt_bbox_train_dir = os.path.join(args.market_path, f"{phase}_id_format")
    else:
        pt_bbox_train_dir = args.out_dir


    if not os.path.exists(pt_bbox_train_dir):
        os.mkdir(pt_bbox_train_dir)

    organize(bbox_dir, pt_bbox_train_dir)

if __name__ == "__main__":
    main()
