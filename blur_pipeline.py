
import torch
from preprocess import KTHBDQDataset
from blur import LearnableGaussian

dataset = KTHBDQDataset(
    root_dir="KTH",
    json_path="kth_clips.json",
    clip_len=32,
    resize=(128, 128)
)

blur = LearnableGaussian()

# Load one preprocessed clip [T, C, H, W]
clip = dataset[0]

# Apply blur frame-by-frame
blurred_clip = torch.stack([
    blur(frame.unsqueeze(0)).squeeze(0)
    for frame in clip
])  # Result shape: [T, C, H, W]
