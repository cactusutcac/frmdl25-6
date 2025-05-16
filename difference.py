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

    def forward(self, bvi):
        if self.bvj is None:
            d = bvi
        else:
            d = bvi - self.bvj
        self.bvj = bvi
        return d