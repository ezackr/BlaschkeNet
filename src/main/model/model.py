import torch
from torch import nn


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
        with torch.no_grad():
            sample_input = torch.zeros(1, 2, num_bins, num_frames)
            sample_output = self.pool1(torch.relu(self.bn1(self.conv1(sample_input))))
            sample_output = self.pool2(torch.relu(self.bn2(self.conv2(sample_output))))
        hidden_dim = int(torch.prod(torch.tensor(sample_output.shape)))
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        # x.shape = (batch_size, num_bins, num_frames, 2)
        x = x.permute(0, 3, 1, 2)
        # x.shape = (batch_size, 2, num_bins, num_frames)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.flatten(self.dropout(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # x.shape = (batch_size, num_classes)
        return torch.softmax(x, dim=1)
