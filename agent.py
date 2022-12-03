import torch 
import torch.nn as nn 

# 4 components 
# 1) convnet for the images 
# 2) fully connected net for measurements 
# 3) fully connected net for command 
# 4) fully connected net for control module


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
        self.fc_dropout = nn.Dropout(p=0.5)

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
    def __init__(self, actions: int, num_commands: int, dropout: float):
        self.perception = PerceptionModule()
        self.measurement = MeasurementsModule(dropout=dropout)
        self.command = CommandModule(num_commands=num_commands, dropout=dropout)
        self.control = ControlModule(actions=actions, dropout=dropout) 

    def forward(self, x):
        
        return x


def main(): 
    num_commands = 4
    actions = 2  # acceleration & steering angle 
    x = torch.randn((1, 3, 200, 88))
    speed = torch.randn((1, 1))
    nav = torch.randn((1, 4))
    print(x.size()) 
    perception = PerceptionModule()
    out = perception(x)
    print(out.size())
    measurements = MeasurementsModule(dropout=0.5)
    out_measure = measurements(speed)
    print(out_measure.size())
    command = CommandModule(num_commands=num_commands, dropout=0.5)
    out_command = command(nav)
    print(out_command.size())

if __name__ == "__main__": 
    main()
