import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock(3, 32, 5, 2, 0.2),
            ConvBlock(32, 32, 3, 1, 0.2),
            ConvBlock(32, 64, 3, 2, 0.2),
            ConvBlock(64, 64, 3, 1, 0.2),
            ConvBlock(64, 128, 3, 2, 0.2),
            ConvBlock(128, 128, 3, 1, 0.2),
            ConvBlock(128, 256, 3, 1, 0.2),
            ConvBlock(256, 256, 3, 1, 0.2),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(in_features=8192, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.act = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc_dropout(self.act(self.fc1(x)))
        x = self.fc_dropout(self.act(self.fc2(x)))
        return x


class MeasurementsModule(nn.Module):
    def __init__(self, dropout: float):
        super(MeasurementsModule, self).__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.drop(self.fc1(x)))
        x = self.act(self.drop(self.fc2(x)))
        return x


class CommandModule(nn.Module):
    def __init__(self, num_commands: int, dropout: float):
        super(CommandModule, self).__init__()
        self.fc1 = nn.Linear(in_features=num_commands, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.drop(self.fc1(x)))
        x = self.act(self.drop(self.fc2(x)))
        return x


class ControlModule(nn.Module):
    def __init__(self, actions: int, dropout: float):
        super(ControlModule, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, actions)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.drop(self.fc1(x)))
        x = self.act(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return x


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.perception = PerceptionModule()
        self.measurement = MeasurementsModule(dropout=cfg.TRAIN.DROPOUT)
        self.command = CommandModule(
            num_commands=cfg.MODEL.NUM_COMMANDS,
            dropout=cfg.TRAIN.DROPOUT,
        )
        self.control = ControlModule(
            actions=cfg.MODEL.NUM_ACTIONS,
            dropout=cfg.TRAIN.DROPOUT,
        )

    def forward(self, img, speed, nav):
        img = self.perception(img)
        speed = self.measurement(speed)
        nav = self.command(nav)

        out = torch.cat((img, speed, nav), dim=1)
        out = self.control(out)

        return out


class BranchedNetwork(nn.Module):
    def __init__(self, cfg):
        super(BranchedNetwork, self).__init__()
        self.perception = PerceptionModule()
        self.measurement = MeasurementsModule(dropout=cfg.TRAIN.DROPOUT)
        self.embedding = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.control_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.Dropout(0.5),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3)
                ) for _ in range(4)
            ]
        )

    def forward(self, img, speed):
        img = self.perception(img)
        speed = self.measurement(speed)
        embed = torch.cat([img, speed], dim=1)
        embed = self.embedding(embed)
        control = torch.cat(
            [branch(embed) for branch in self.control_branches], dim=1
        )
        return control
