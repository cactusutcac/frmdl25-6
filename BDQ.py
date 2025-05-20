from blur import LearnableGaussian
from difference import Difference
from quantization import DifferentiableQuantization
from torch import nn

class BDQEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            LearnableGaussian(),
            Difference(),
            DifferentiableQuantization(),
        )

    """
    Args:
        x: the input tensor (video frame)
    """
    def forward(self, x):
        for layer in self.encoder:
            x = layer.forward(x)
        
        return x