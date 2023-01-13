import os
import cv2 
import glob 
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader 
import albumentations as A 
from albumentations.pytorch import ToTensorV2


def get_train_transform(): 
    return A.Compose(
        [
            A.Normalize(
                mean=(0.2543, 0.2641, 0.2714), 
                std=(0.1730, 0.1726, 0.1726)
            ), 
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
        
        if self.transform is not None: 
            image = self.transform(image=img)['image']

        target = torch.tensor(target) 
        return image, target  


def main():
    root_dir = "/data/AgentHuman/SeqTrain" 
    imgs_dir = os.path.join(root_dir, "images") 
    targets_dir = os.path.join(root_dir, "targets")
    dataset = ImitationLearningDataset(imgs_dir, targets_dir, transform=get_train_transform()) 
    # dataset = ImitationLearningDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=1)

    for img, target in dataloader: 
        print(img.shape)
        print(target.shape)
        break

if __name__ == "__main__":
    main()