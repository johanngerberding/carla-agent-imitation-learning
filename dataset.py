import os
import cv2 
import glob 
import numpy as np
import torchvision 
from torch.utils.data import Dataset, DataLoader 
import albumentations as A 
from albumentations.pytorch import ToTensorV2

from utils import mean_std


def get_train_transform(): 
    return A.Compose(
        [
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1, 1, 1)), 
            ToTensorV2(),
        ]
    )



class ImitationLearningDataset(Dataset):
    """CARLA Imitation Learning Dataset"""  
    def __init__(self, imgs_dir: str, targets_dir: str, transform = None): 
        self.imgs = glob.glob(imgs_dir + "/*.png")
        self.targets = glob.glob(targets_dir + "/*.txt")
        self.transform = transform  
        

    def __len__(self): 
        return len(self.imgs) 

    def __getitem__(self, idx: int): 
        target = np.loadtxt(self.targets[idx]) 
        img = cv2.imread(self.imgs[idx]) 
        # normalize 
        # to tensor  
        if self.transform is not None: 
            image = self.transform(image=img)['image'] 
        return image  


def main():
    root_dir = "/data/AgentHuman/SeqTrain" 
    imgs_dir = os.path.join(root_dir, "images") 
    targets_dir = os.path.join(root_dir, "targets")
    dataset = ImitationLearningDataset(imgs_dir, targets_dir, transform=get_train_transform()) 
    # dataset = ImitationLearningDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=1)

    mean, std = mean_std(dataloader, 4)
    print(mean)
    print(std)

if __name__ == "__main__":
    main()