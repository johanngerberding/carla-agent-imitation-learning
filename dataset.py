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
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.CLAHE(p=0.2),
            A.OneOf([
                A.RandomRain(),
                A.RandomFog(),
                A.Solarize(),
                A.RandomSnow(),
            ], p=0.3),
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
        self.num_classes = cfg.MODEL.NUM_COMMANDS
        self.branched = cfg.MODEL.BRANCHED

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        target = np.loadtxt(self.targets[idx])
        org_image = cv2.imread(self.imgs[idx])
        if self.transform is not None:
            image = self.transform(image=org_image)['image']

        target = torch.tensor(target)

        # there are some 0 values, bit hacky and not sure if correct
        if target[24].long() == 0:
            target[24] = 2

        command = target[24].long() - 2

        cmd = F.one_hot(
            command,
            num_classes=self.num_classes
        )
        speed = target[10].unsqueeze(0).float()

        if self.branched:
            mask = torch.zeros((4, 3))
            mask[command, :] = 1
            target = torch.stack([target[:3] for _ in range(4)], dim=0)
            return org_image, image, speed, mask.reshape(-1), target.reshape(-1)

        target = target[:3].float()
        return org_image, image, speed, cmd.float(), target
