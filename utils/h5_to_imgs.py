import argparse
import os
import glob
import tqdm
import cv2
import h5py
import numpy as np


def h5_to_imgs(root_dir: str, imgs_dir: str = None, targets_dir: str = None):
    """Extract training images and targets from h5 files.

    Args:
        root_dir (str): dataset root directory
        imgs_dir (str, optional): Where to save images. Defaults to None.
        targets_dir (str, optional): where to save targets. Defaults to None.
    """
    files = glob.glob(root_dir + "/*.h5")
    idx = 1

    if imgs_dir is None:
        imgs_dir = os.path.join(root_dir, "images")
        if not os.path.isdir(imgs_dir):
            os.mkdir(imgs_dir)
            print(f"Generated images dir: {imgs_dir}")
    if targets_dir is None:
        targets_dir = os.path.join(root_dir, "targets")
        if not os.path.isdir(targets_dir):
            os.mkdir(targets_dir)
            print(f"Generated targets dir: {targets_dir}")

    for fi in tqdm.tqdm(files):
        try:
            f = h5py.File(fi, 'r')
            imgs = f['rgb']
            targets = f['targets']
            for i in range(imgs.shape[0]):
                img = imgs[i]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                target = targets[i]
                filename = str(idx).zfill(10) + ".png"
                targetname = str(idx).zfill(10) + ".txt"
                idx += 1
                cv2.imwrite(os.path.join(imgs_dir, filename), img)
                np.savetxt(os.path.join(targets_dir, targetname), target)
        except Exception:
            print(f"Could not read: {fi}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Path to AgentHuman folder.")
    args = parser.parse_args()
    print("Extract training data...")
    h5_to_imgs(os.path.join(args.dir, "SeqTrain"))
    print("Extract validation data ...")
    h5_to_imgs(os.path.join(args.dir, "SeqVal"))


if __name__ == "__main__":
    main()
