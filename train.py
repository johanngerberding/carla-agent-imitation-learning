import os
import glob
import cv2
import torch
import tqdm
import wandb
from torch.utils.data import DataLoader
from dataset import ImitationLearningDataset
from agent import Network

from config import get_cfg_defaults


def train_epoch(model, dataloader, loss_fn, optimizer, device, cfg):
    model.train()
    losses = []
    for idx, (img, speed, nav, target) in enumerate(dataloader):
        img = img.to(device)
        speed = speed.to(device)
        nav = nav.to(device)
        target = target.to(device)
        out = model(img, speed, nav)
        loss = loss_fn(out, target)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        if (idx + 1) % cfg.TRAIN.PRINT_INTERVAL == 0 and idx != 0:
            print(f"Loss Iteration {idx + 1}: {loss.item()}")


def eval_epoch(model, dataloader, loss_fn, device, cfg):
    model.eval()
    for idx, (img, speed, nav, target) in enumerate(dataloader):
        img = img.to(device)
        speed = speed.to(device)
        nav = nav.to(device)
        target = target.to(device)

        with torch.no_grad():
            out = model(img, speed, nav)
        loss = loss_fn(out, target)

        if (idx + 1) % cfg.VAL.PRINT_INTERVAL and idx != 0:
            print(f"Loss Iteration {idx + 1}: {loss.item()}")



def main():
    opts = None  # add argparser and function to extract config stuff
    cfg = get_cfg_defaults()
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()

    wandb_args = {
        "epochs": cfg.TRAIN.NUM_EPOCHS,
        "initial_lr": cfg.TRAIN.LR,
        "train_batch_size": cfg.TRAIN.BATCH_SIZE,
        "optimizer": cfg.TRAIN.OPTIM,
    }
    wandb.init(config=wandb_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImitationLearningDataset(cfg, "train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
    )

    val_dataset = ImitationLearningDataset(cfg, "val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
    )

    model = Network(cfg)
    model.to(device)
    model.train()
    wandb.watch(model, log_freq=100)

    if cfg.TRAIN.OPTIM == "adam":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.TRAIN.LR,
        )
    else:
        raise NotImplementedError("This optimizer is not implemented")

    loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        eval_epoch(model, val_dataloader, loss_fn, device)


if __name__ == "__main__":
    main()
