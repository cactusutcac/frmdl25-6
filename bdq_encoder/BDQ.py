from bdq_encoder.blur import LearnableGaussian
from bdq_encoder.difference import Difference
from bdq_encoder.quantization import DifferentiableQuantization
from torch import nn

class BDQEncoder(nn.Module):
    """
    Sequentially combines the blur, difference and quantization parts
    to form the BDQ encoder. 
    """
    def __init__(self, hardness=5.0):
        super().__init__()
        self.encoder = nn.Sequential(
            LearnableGaussian(),
            Difference(),
            DifferentiableQuantization(hardness=hardness),
        )

    def forward(self, x):
        """
        Args:
            x: the input tensor (video frame).
        """
        for layer in self.encoder:
            x = layer.forward(x)
        
        return x