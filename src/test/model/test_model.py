import torch

from src.main.model import BlaschkeNet


def test_model():
    batch_size = 32
    num_bins = 245
    num_frames = 65
    num_classes = 3
    model = BlaschkeNet(num_bins, num_frames, num_classes)
    x = torch.ones(size=(batch_size, num_bins, num_frames, 2))
    y = model(x)
    assert y.shape == (batch_size, num_classes)
