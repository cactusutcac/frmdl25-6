# import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# from torchsummary import summary
# from PIL import Image

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