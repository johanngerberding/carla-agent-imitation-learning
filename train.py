import os 
import glob 
import cv2 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from dataset import ImitationLearningDataset, get_train_transform
from agent import Network


def main():
    batch_size = 8  
    num_workers = 4 
    root_dir = "/data/AgentHuman/SeqTrain"
    imgs_dir = os.path.join(root_dir, "images")
    targets_dir = os.path.join(root_dir, "targets")

    dataset = ImitationLearningDataset(imgs_dir, targets_dir, transform=get_train_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    model = Network() 


if __name__ == "__main__":
    main()