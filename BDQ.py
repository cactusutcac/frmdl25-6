from blur import LearnableGaussian
from difference import Difference
from quantization import DifferentiableQuantization
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
    
    def freeze(self):
        """
        Freezes the parameters to prevent/pause learning. 
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Resumes learning for BDQ encoder parameters.
        """
        for param in self.parameters():
            param.requires_grad = True