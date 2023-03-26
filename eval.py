import os
import torch
from dataset import ImitationLearningDataset
from config import get_cfg_defaults
from torch.utils.data import DataLoader
from utils import AverageMeter


def main():
    exp_dir = "/home/johann/dev/conditional-imitation-learning-pytorch/exps/2023-03-23"
    best_checkpoint = os.path.join("checkpoints/best.pth")
    exp_cfg = os.path.join(exp_dir, "config.yaml")
    cfg = get_cfg_defaults()

    if os.path.isfile(exp_cfg):
        cfg.merge_from_file(exp_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = ImitationLearningDataset(cfg, "val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        shuffle=True,
    )

    model = torch.load(best_checkpoint)
    model = model.float()
    model.to(device)
    model.eval()

    if cfg.MODEL.BRANCHED:
        loss_fn = torch.nn.MSELoss(reduction="sum")
    else:
        loss_fn = torch.nn.MSELoss(reduction="mean")

    loss_weights = {
        'steer': 0.5,
        'acc': 0.45,
        'brake': 0.05,
    }
    val_losses = AverageMeter()

    for idx, (org_img, img, speed, nav_mask, target) in enumerate(val_dataloader):
        img = img.to(device)
        speed = speed.to(device)
        nav_mask = nav_mask.to(device)
        target = target.to(device)

        if cfg.MODEL.BRANCHED:
            with torch.no_grad():
                out = model(img, speed)
            out = out * nav_mask
            target = target * nav_mask
            steer_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, :, 0],
                target.reshape((-1, 4, 3))[:, :, 0],
            )
            acc_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, :, 1],
                target.reshape((-1, 4, 3))[:, :, 1],
            )
            brake_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, :, 2],
                target.reshape((-1, 4, 3))[:, :, 2],
            )
            loss = (
                loss_weights['steer'] * steer_loss +
                loss_weights['acc'] * acc_loss +
                loss_weights['brake'] * brake_loss
            )
        else:
            with torch.no_grad():
                out = model(img, speed, nav_mask)

            steer_loss = loss_fn(out[:, 0], target[:, 0])
            acc_loss = loss_fn(out[:, 1], target[:, 1])
            brake_loss = loss_fn(out[:, 2], target[:, 2])
            loss = (
                loss_weights['steer'] * steer_loss +
                loss_weights['acc'] * acc_loss +
                loss_weights['brake'] * brake_loss
            )

        val_losses.update(loss.item(), cfg.VAL.BATCH_SIZE)

    print(f"Average validation loss: {val_losses.avg}")


if __name__ == "__main__":
    main()
