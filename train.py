import os
import torch
import shutil
# import wandb
from torch.utils.data import DataLoader
from torch.nn import Module
from dataset import ImitationLearningDataset
from model import Network, BranchedNetwork
from torch.utils.tensorboard import SummaryWriter
from config import get_cfg_defaults
from datetime import datetime


def train_epoch(
    model: Module,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    device,
    cfg,
    writer: SummaryWriter,
    epoch: int,
):
    model.train()
    for idx, (org_img, img, speed, nav_mask, target) in enumerate(dataloader):
        n_iter = epoch * len(dataloader) + idx + 1
        img = img.to(device)
        speed = speed.to(device)
        nav_mask = nav_mask.to(device)
        target = target.to(device)

        if cfg.MODEL.BRANCHED:
            out = model(img, speed)
            target = target * nav_mask
            out = out * nav_mask
            steer_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, 0],
                target.reshape((-1, 4, 3))[:, 0],
            )
            acc_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, 1],
                target.reshape((-1, 4, 3))[:, 1],
            )
            brake_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, 2],
                target.reshape((-1, 4, 3))[:, 2],
            )
            loss = steer_loss + 0.5 * (acc_loss + brake_loss)
            loss.backward()
        else:
            out = model(img, speed, nav_mask)
            steer_loss = loss_fn(out[:, 0], target[:, 0])
            acc_loss = loss_fn(out[:, 1], target[:, 1])
            brake_loss = loss_fn(out[:, 2], target[:, 2])
            loss = steer_loss + 0.5 * (acc_loss + brake_loss)
            loss.backward()

        optimizer.step()
        if (idx + 1) % cfg.TRAIN.PRINT_INTERVAL == 0 and idx != 0:
            print(f"train: epoch {epoch + 1} iteration {n_iter} loss: {loss.item()}")
            # wandb.log({"train_loss": loss.item()})
            writer.add_scalar('MSELoss/train', loss.item(), n_iter)
            writer.add_scalar('MSELoss_steer/train', steer_loss.item(), n_iter)
            writer.add_scalar('MSELoss_acc/train', acc_loss.item(), n_iter)
            writer.add_scalar('MSELoss_brake/train', brake_loss.item(), n_iter)


def eval_epoch(
    model: Module,
    dataloader: DataLoader,
    loss_fn,
    device,
    cfg,
    writer: SummaryWriter,
    epoch: int
):
    model.eval()
    for idx, (org_img, img, speed, nav_mask, target) in enumerate(dataloader):
        n_iter = epoch * len(dataloader) + idx + 1
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
                out.reshape((-1, 4, 3))[:, 0],
                target.reshape((-1, 4, 3))[:, 0],
            )
            acc_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, 1],
                target.reshape((-1, 4, 3))[:, 1],
            )
            brake_loss = loss_fn(
                out.reshape((-1, 4, 3))[:, 2],
                target.reshape((-1, 4, 3))[:, 2],
            )
            loss = steer_loss + 0.5 * (acc_loss + brake_loss)

        else:
            with torch.no_grad():
                out = model(img, speed, nav_mask)

            steer_loss = loss_fn(out[:, 0], target[:, 0])
            acc_loss = loss_fn(out[:, 1], target[:, 1])
            brake_loss = loss_fn(out[:, 2], target[:, 2])
            loss = steer_loss + 0.5 * (acc_loss + brake_loss)

        if (idx + 1) % cfg.VAL.PRINT_INTERVAL == 0 and idx != 0:
            print(f"val: epoch {epoch + 1} iteration {n_iter} loss: {loss.item()}")
            # wandb.log({"val_loss": loss.item()})
            writer.add_scalar('MSELoss/val', loss.item(), n_iter)
            writer.add_scalar('MSELoss_steer/val', steer_loss.item(), n_iter)
            writer.add_scalar('MSELoss_acc/val', acc_loss.item(), n_iter)
            writer.add_scalar('MSELoss_brake/val', brake_loss.item(), n_iter)


def main():
    opts = None  # add argparser and function to extract config stuff
    cfg = get_cfg_defaults()
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    writer = SummaryWriter()
    td = datetime.now().strftime("%Y-%m-%d")
    exp_dir = f"exps/{td}"
    exp_dir = os.path.join(os.path.abspath(os.getcwd()), exp_dir)
    print(f"Experiment folder: {exp_dir}")
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    else:
        shutil.rmtree(exp_dir)

    ckpts_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpts_dir):
        os.makedirs(ckpts_dir)
    """
    wandb_args = {
        "epochs": cfg.TRAIN.NUM_EPOCHS,
        "initial_lr": cfg.TRAIN.LR,
        "train_batch_size": cfg.TRAIN.BATCH_SIZE,
        "optimizer": cfg.TRAIN.OPTIM,
    }
    wandb.init(config=wandb_args)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ImitationLearningDataset(cfg, "train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        shuffle=True,
    )

    val_dataset = ImitationLearningDataset(cfg, "val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        num_workers=cfg.SYSTEM.NUM_WORKERS,
        shuffle=True,
    )

    if cfg.MODEL.BRANCHED:
        model = BranchedNetwork(cfg)
    else:
        model = Network(cfg)

    model.to(device)
    model.train()
    # wandb.watch(model, log_freq=100)

    if cfg.TRAIN.OPTIM == "adam":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.TRAIN.LR,
        )
    else:
        raise NotImplementedError("This optimizer is not implemented")

    loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        train_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device,
            cfg,
            writer,
            epoch,
        )
        eval_epoch(model, val_dataloader, loss_fn, device, cfg, writer, epoch)
        checkp = os.path.join(ckpts_dir, f"epoch-{str(epoch + 1).zfill(3)}.pth")
        torch.save(model, checkp)
        print(f"Saved checkpoint: {checkp}")


if __name__ == "__main__":
    main()
