import os 
import glob 
import cv2 
import torch 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from dataset import ImitationLearningDataset, get_train_transform
from agent import Network


def main():
    batch_size = 8  
    num_workers = 4 
    num_actions = 3 
    num_commands = 4 
    dropout = 0.5 
    root_dir = "/data/AgentHuman/SeqTrain"
    imgs_dir = os.path.join(root_dir, "images")
    targets_dir = os.path.join(root_dir, "targets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImitationLearningDataset(imgs_dir, targets_dir, transform=get_train_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    model = Network(actions=num_actions, num_commands=num_commands, dropout=dropout) 
    model.to(device)

    for img, speed, nav, target in dataloader: 
        img = img.to(device)
        speed = speed.to(device)
        nav = nav.to(device)
        out = model(img, speed, nav)
        print(out.shape)

        break



if __name__ == "__main__":
    main()