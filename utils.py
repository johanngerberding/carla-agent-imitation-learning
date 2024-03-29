import torch
import tqdm
from torch.utils.data import DataLoader
from config import get_cfg_defaults

cfg = get_cfg_defaults()


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
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    cnt = len(dataloader) * bs * cfg.DATA.IMG_HEIGHT * cfg.DATA.IMG_WIDTH
    # mean and std
    mean = psum / cnt
    total_var = (psum_sq / cnt) - (mean ** 2)
    std = torch.sqrt(total_var)

    return mean, std


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
