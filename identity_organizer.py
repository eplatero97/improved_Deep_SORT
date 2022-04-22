import os
import sys
import shutil

def organize(src: str, dest: str):
    for file in os.listdir(src):
        identity = file.split("_")[0]
        identity_dir = os.path.join(dest, identity)
        if not os.path.exists(identity_dir):
            os.mkdir(identity_dir)
        if file not in os.listdir(identity_dir):
            file_path = os.path.join(src, file)
            shutil.copy(file_path, identity_dir)

if len(sys.argv) != 2:
    print("Usage: \"python identity_organizer.py <PATH TO MARKET1501>\"")
    exit()

if os.path.exists(sys.argv[1]):
    market_dir = sys.argv[1]

elif os.path.exists(os.path.join(os.getcwd, sys.argv[1])):
    market_dir = os.path.join(os.getcwd, sys.argv[1])

else:
    print("Invalid Path")
    exit()

bbox_train_dir = os.path.join(market_dir, "bounding_box_train")
gt_bbox_dir = os.path.join(market_dir, "gt_bbox")
pt_bbox_train_dir = os.path.join(market_dir, "bounding_box_train_pt_format")
pt_gt_bbox_dir = os.path.join(market_dir, "gt_bbox_pt_format")

if not os.path.exists(pt_bbox_train_dir):
    os.mkdir(pt_bbox_train_dir)

if not os.path.exists(pt_gt_bbox_dir):
    os.mkdir(pt_gt_bbox_dir)

organize(bbox_train_dir, pt_bbox_train_dir)
organize(gt_bbox_dir, pt_gt_bbox_dir)


