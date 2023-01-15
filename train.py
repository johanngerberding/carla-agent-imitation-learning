import torch
# import wandb
from torch.utils.data import DataLoader
from dataset import ImitationLearningDataset
from model import Network
from torch.utils.tensorboard import SummaryWriter
from config import get_cfg_defaults


def train_epoch(model, dataloader, loss_fn, optimizer, device, cfg, writer, epoch):
    model.train()
    for idx, (img, speed, nav, target) in enumerate(dataloader):
        n_iter = epoch * len(dataloader) + idx + 1
        img = img.to(device)
        speed = speed.to(device)
        nav = nav.to(device)
        target = target.to(device)
        out = model(img, speed, nav)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        if (idx + 1) % cfg.TRAIN.PRINT_INTERVAL == 0 and idx != 0:
            print(f"epoch {epoch + 1} iteration {idx + 1} loss: {loss.item()}")
            # wandb.log({"train_loss": loss.item()})
            writer.add_scalar('Loss/train', loss.item(), n_iter)


def eval_epoch(model, dataloader, loss_fn, device, cfg, writer, epoch):
    model.eval()
    for idx, (img, speed, nav, target) in enumerate(dataloader):
        n_iter = epoch * len(dataloader) + idx + 1
        img = img.to(device)
        speed = speed.to(device)
        nav = nav.to(device)
        target = target.to(device)

        with torch.no_grad():
            out = model(img, speed, nav)
        loss = loss_fn(out, target)

        if (idx + 1) % cfg.VAL.PRINT_INTERVAL == 0 and idx != 0:
            print(f"epoch {epoch + 1} iteration {n_iter} loss: {loss.item()}")
            # wandb.log({"val_loss": loss.item()})
            writer.add_scalar('Loss/val', loss.item(), n_iter)


def main():
    opts = None  # add argparser and function to extract config stuff
    cfg = get_cfg_defaults()
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    writer = SummaryWriter()
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
        train_epoch(model, train_dataloader, loss_fn, optimizer, device, cfg, writer, epoch)
        eval_epoch(model, val_dataloader, loss_fn, device, cfg, writer, epoch)


if __name__ == "__main__":
    main()
