import os 
import glob 
import h5py
import cv2 
import tqdm 
import numpy as np 
import torch
from torch.utils.data import DataLoader


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
        except: 
            print(f"Could not read: {fi}") 


def mean_std(dataloader: DataLoader, bs: int) -> tuple:
    """Helper function to calculate mean and std of dataset.

    Args:
        dataloader (DataLoader): Imitation Learning dataset dataloader 
        bs (int): batch size 

    Returns:
        tuple: mean, std 
    """
    psum = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    psum_sq = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    # loop through images
    for inputs in tqdm.tqdm(dataloader):
        psum += inputs.sum(axis = [0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis = [0, 2, 3]) 
    
    cnt = len(dataloader) * bs * 88 * 200 
    # mean and std
    mean = psum / cnt
    total_var  = (psum_sq / cnt) - (mean ** 2)
    std  = torch.sqrt(total_var)  
    
    return mean, std          
