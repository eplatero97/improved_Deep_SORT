from jsonargparse import ArgumentParser, ActionConfigFile
from io import TextIOWrapper
from PIL import Image
from pathlib import Path
import sys
import os

def crop(fname: str, framesPath: str, destPath: str):
    with open(fname, 'r') as file:
        lines = file.readlines()
        for line in lines:
            args = line.split(',')
            frameNum = args[0]
            id = args[1].zfill(3)
            x = int(args[2])
            y = int(args[3])
            width = int(args[4])
            height = int(args[5])
            idPath = os.path.join(destPath, id)
            if (not os.path.exists(idPath)):
                os.mkdir(idPath)
            frame = Image.open(os.path.join(framesPath, frameNum.zfill(6) + ".jpg"))
            croppedImg = frame.crop((x, y, x + width, y + height))
            out_img = os.path.join(idPath, frameNum + "_" + id + ".jpg")
            croppedImg.save(out_img)


def main(mot_path: str, is_det: bool):
    mot_path = Path(mot_path)
    if is_det:
        mot_files = mot_path.rglob("det.txt")
        cropped_dir_name = "crops_det"
    else:
        mot_files = mot_path.rglob("gt.txt")
        cropped_dir_name = "crops_gt"
    
    for mot_file in mot_files:
        video_path = mot_file.parent.parent
        frames_path = video_path / "img1"
        cropped_path = video_path / cropped_dir_name
        cropped_path.mkdir(exist_ok = True)
        crop(mot_file, frames_path, cropped_path)
        
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mot_path", type=str, required=True, help="path to MOT directory in local machine")
    parser.add_argument("--is_det", type=bool, required=False, help="do we use bboxes in `det.txt`?")
    parser.add_argument("--config", action = ActionConfigFile)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.mot_path, args.is_det)