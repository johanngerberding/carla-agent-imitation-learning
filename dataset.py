import os
import cv2 
import glob 
import numpy as np
import torch
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader 
import albumentations as A 
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt 


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
    def __init__(self, imgs_dir: str, targets_dir: str, transform = None, num_classes: int = 4): 
        self.imgs = glob.glob(imgs_dir + "/*.png")
        self.targets = glob.glob(targets_dir + "/*.txt")
        self.transform = transform 
        self.num_classes = num_classes 
        

    def __len__(self): 
        return len(self.imgs) 

    def __getitem__(self, idx: int): 
        target = np.loadtxt(self.targets[idx]) 
        img = cv2.imread(self.imgs[idx]) 
        if self.transform is not None: 
            image = self.transform(image=img)['image']

        target = torch.tensor(target)

        # there are some 0 values, bit hacky and not sure if correct 
        if target[24].long() == 0: 
            target[24] = 2

        command = F.one_hot(target[24].long() - 2, num_classes=self.num_classes)
        speed = target[10].unsqueeze(0).float() 

        target = target[:3].float()

        return image, speed, command.float(), target 


def main():
    root_dir = "/data/AgentHuman/SeqTrain" 
    imgs_dir = os.path.join(root_dir, "images") 
    targets_dir = os.path.join(root_dir, "targets")
   
    dataset = ImitationLearningDataset(imgs_dir, targets_dir, transform=get_train_transform()) 
    # dataset = ImitationLearningDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=1)

    for img, speed, nav, target  in dataloader: 
        print(img.shape)
        print(speed)  
        print(nav)
        print(target) 
        break


if __name__ == "__main__":
    main()