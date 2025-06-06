import torch
import torch.nn as nn

class Difference(nn.Module):
    def __init__(self):
        super(Difference, self).__init__()
        self.bvj = None

    """
    Args:
        x: the input frames tensor of shape (B, T, C, H, W), i.e. video with T frames
    """
    def forward(self, x):
        d = x.roll(-1, dims=1) - x
        return d