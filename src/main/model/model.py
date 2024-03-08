import torch
from torch import nn

num_hidden_channels: int = 32


class BlaschkeNet(nn.Module):
    def __init__(
            self,
            num_bins: int,
            num_frames: int,
            num_classes: int = 3,
            kernel_size: int = 3,
            p: float = 0.25
    ):
        super(BlaschkeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=p)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (num_bins // 4) * (num_frames // 4), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(self.dropout(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
