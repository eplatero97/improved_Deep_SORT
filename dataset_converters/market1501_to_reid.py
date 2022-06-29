import os
import shutil
from jsonargparse import ArgumentParser, ActionConfigFile
from pathlib import Path
from contextlib import redirect_stdout
from typing import List, Tuple
from loguru import logger # TODO: use `loguru` to create any debugging errors
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--market_path", type=str, required=True, help="Path to Market-1501 dataset")
    parser.add_argument("--out_dir", type=str, required=False, help="Output path to store human identities", default=None)
    parser.add_argument("--config", action = ActionConfigFile)
    args = parser.parse_args()
    return args

def extract_images(market_path: Path) -> Tuple[list, list]:
    """
    :obj: extract file paths of images in train and test dir
    :param market_path: path to `market1501` dataset dir
    :return: lists that contain paths of train and test images
    """
    train_imgs: List[Path] = sorted(list((market_path / "bounding_box_train").rglob("*.jpg")))
    test_imgs: List[Path] = sorted(list((market_path / "bounding_box_test").rglob("*.jpg")))
    filtered_test_imgs: List[Path] = [test_img for test_img in test_imgs if test_img.name.split('_')[0] != "-1"] # filter out all imgs with "-1" id
    logger.info(f"single training image path: {train_imgs[0]}")
    logger.info(f"single testing image path: {filtered_test_imgs[0]}")
    return train_imgs, filtered_test_imgs

def create_ids(dest: Path, imgs: List[Path]) -> None:
    """
    :obj: create a directory per unique id and store all images with its corresponding id
    :param market_path: path to `market1501` dataset dir
    :param dest: directory path of where to create id directories
    :imgs: list containig paths of training and testing images
    :n_train: number of training images
    """
    logger.info(f"`imgs` dir path: {dest}")
    for file in tqdm(imgs):
        
        id: str = file.name.split('_')[0] # what is the id of the image?
        id_dir: Path = dest / id # create id directory
        id_dir.mkdir(parents=True, exist_ok=True) # recursively create dir if it does not exist

        # is the current file NOT a child file of `id_dir`?
        children_files: List[str] = list(id_dir.iterdir())
        if file not in children_files:
            new_file_path = id_dir / file.name
            shutil.copy(file, new_file_path)

def create_meta_file(phase_imgs: List[str], out_file: str) -> None:
    """
    :obj: create metafile 
    :param phase_imgs: list that contains paths of images in train or test phase
    :param out_file: name of output file
    """
    # make each entry in `meta_contents` follow the pattern: {id}/{file} {id}
    meta_contents: List[str] = []
    for file in phase_imgs:
        id: str = file.name.split('_')[0]
        content: str = f"{id}/{file.name} {id}"
        meta_contents.append(content)
    
    # make a new line for each entry in `meta_contents`
    with open(out_file, 'w') as out:
        with redirect_stdout(out):
            for content in meta_contents:
                print(content)


def organize(market_path: Path, dest: Path) -> None:
    """
    :obj: transform train and test images of Market-1501 dataset to re-id forat of mmtracking
    :param market_path: path to `market1501` dataset dir
    :param dest: path to store dataset in re-id format of mmtracking
    """

    train_imgs, test_imgs = extract_images(market_path)
    n_train: int = len(train_imgs)
    n_test: int = len(test_imgs)
    n_imgs: int = n_train+n_test
    train_ratio: float = round(n_train / n_imgs, 2)
    
    imgs: List[str] = train_imgs + test_imgs
    ids_path = dest / "imgs" # path to store id dirs
    create_ids(ids_path, imgs)
    
    meta_path: Path = dest / "meta"
    meta_path.mkdir(exist_ok=True)

    train_meta_file = meta_path / f"train_{train_ratio*100:.0f}.txt"
    val_meta_file = meta_path / f"val_{(1-train_ratio)*100:.0f}.txt"

    create_meta_file(train_imgs, train_meta_file)
    create_meta_file(test_imgs, val_meta_file)

def main():
    args = parse_args()
    market_path = Path(args.market_path)
    if args.out_dir is None:
        out_dir = market_path / "mmReIdFormat"
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(exist_ok=True)

    logger.info(f"market-1501 dataset path: {market_path}")
    logger.info(f"mmreid path: {out_dir}")

    organize(market_path, out_dir)

if __name__ == "__main__":
    main()
