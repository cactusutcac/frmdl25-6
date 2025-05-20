import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

class LearnableGaussian(nn.Module):
    def __init__(self, kernel_size=5, init_sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor(init_sigma))

    def forward(self, x):
        # Make sure sigma is positive
        sigma = self.sigma

        # Create 1D kernel
        k = self.kernel_size // 2
        coords = torch.arange(-k, k + 1, dtype=torch.float32, device=x.device)
        gauss = torch.exp(-0.5 * (coords / sigma)**2)
        gauss /= gauss.sum()

        # Make 2D kernel
        kernel = torch.outer(gauss, gauss)
        kernel = kernel.expand(x.shape[1], 1, self.kernel_size, self.kernel_size)

        # Apply depthwise convolution
        return F.conv2d(x, kernel, padding=k, groups=x.shape[1])

# Load image
img = Image.open("test.jpg").convert("RGB")
img_tensor = ToTensor()(img).unsqueeze(0)  # (1, C, H, W)

# Apply blur
model = LearnableGaussian()
output = model(img_tensor)

# Save result
ToPILImage()(output.squeeze(0).clamp(0, 1)).save("gaussian_learned.jpg")
