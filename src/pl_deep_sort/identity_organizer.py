import os
import shutil
from jsonargparse import ArgumentParser, ActionConfigFile
from pathlib import Path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to Market-1501 dataset")
    parser.add_argument("--train_output", type=str, required=False, help="Output path for bounding_box_train identities", default=None)
    parser.add_argument("--gt_output", type=str, required=False, help="Output path for gt_bbox identities", default=None)
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

args = parse_args()

bbox_train_dir = os.path.join(args.data_path, "bounding_box_train")
gt_bbox_dir = os.path.join(args.data_path, "gt_bbox")
if args.train_output == None:
    pt_bbox_train_dir = os.path.join(args.data_path, "bounding_box_train_pt_format")
else:
    pt_bbox_train_dir = args.train_output
if args.gt_output == None:
    pt_gt_bbox_dir = os.path.join(args.data_path, "gt_bbox_pt_format")
else:
    pt_gt_bbox_dir = args.gt_output

if not os.path.exists(pt_bbox_train_dir):
    os.mkdir(pt_bbox_train_dir)

if not os.path.exists(pt_gt_bbox_dir):
    os.mkdir(pt_gt_bbox_dir)

organize(bbox_train_dir, pt_bbox_train_dir)
organize(gt_bbox_dir, pt_gt_bbox_dir)
