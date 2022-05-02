from argparse import ArgumentParser
from torchvision.datasets import ImageFolder
from PIL import Image
import csv
import os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to Market-1501 format dataset")
    parser.add_argument("--output_path", type=str, required=False, help="Output folder of CSV")
    parser.add_argument("--output_name", type=str, required=False, help="Output filename of CSV")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dims = [Image.open(img[0]).size for img in ImageFolder(args.data_path).imgs]

    if args.output_path != None and not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if args.output_path != None and args.output_name != None:
        output_file = open(f"{args.output_path}/{args.output_name}.csv", 'w')
    elif args.output_name != None:
        output_file = open(f"{args.output_name}.csv", 'w')
    elif args.output_path != None:
        output_file = open(f"{args.output_path}/dimensions.csv", 'w')
    else:
        output_file = open("dimensions.csv", 'w')
    
    csv_out = csv.writer(output_file)
    csv_out.writerow(["width", "height"])
    for dim in dims:
        csv_out.writerow(dim)

if __name__ == "__main__":
    main()