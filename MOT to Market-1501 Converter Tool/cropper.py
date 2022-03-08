from io import TextIOWrapper
from PIL import Image
import sys
import os

def crop(bounds: TextIOWrapper, video, framesPath, destPath):
    for line in bounds:
        args = line.split(',')
        frameNum = args[0]
        id = args[1]
        x = int(args[2])
        y = int(args[3])
        width = int(args[4])
        height = int(args[5])
        idPath = os.path.join(destPath, video + "_" + id)
        if (not os.path.exists(idPath)):
            os.mkdir(idPath)
        frame = Image.open(os.path.join(framesPath, frameNum.zfill(6) + ".jpg"))
        croppedImg = frame.crop((x, y, x + width, y + height))
        croppedImg = croppedImg.save(os.path.join(idPath, video + "_" + frameNum + "_" + id + ".jpg"))
        
        


def main():
    if os.path.exists(os.path.join(os.getcwd(), sys.argv[1])):
        directory = os.path.join(os.getcwd(), sys.argv[1])
    elif os.path.exists(sys.argv[1]):
        directory = sys.argv[1]
    else:
        print("Invalid Path")
        return -1

    cropsPath = os.path.join(directory, "crops")
    if (not os.path.exists(cropsPath)):
        os.mkdir(cropsPath)
        os.mkdir(os.path.join(cropsPath, "gt"))
    for video in os.listdir(directory):
        if (video == "crops"):
            continue
        videoPath = os.path.join(directory, video)
        framesPath = os.path.join(videoPath, "img1")
        if (os.path.exists(os.path.join(videoPath, "gt"))):
            crop(open(os.path.join(videoPath, "gt/gt.txt")), video, framesPath, os.path.join(cropsPath, "gt"))
        

if __name__ == "__main__":
    main()