import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
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


def get_val_transform():
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
    def __init__(
        self,
        cfg,
        mode: str,
        num_classes: int = 4
    ):
        if mode == "train":
            imgs_dir = cfg.DATA.TRAIN_IMGS_DIR
            targets_dir = cfg.DATA.TRAIN_TARGETS_DIR
        else:
            imgs_dir = cfg.DATA.VAL_IMGS_DIR
            targets_dir = cfg.DATA.VAL_TARGETS_DIR

        self.imgs = glob.glob(imgs_dir + "/*.png")
        self.targets = glob.glob(targets_dir + "/*.txt")
        self.transform = (
            get_train_transform() if mode == "train" else get_val_transform()
        )
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

        command = F.one_hot(
            target[24].long() - 2,
            num_classes=self.num_classes
        )
        speed = target[10].unsqueeze(0).float()

        target = target[:3].float()

        return image, speed, command.float(), target
