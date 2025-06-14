from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os
import random
import numpy as np
import torch
from tqdm import tqdm

from bdq_encoder.BDQ import BDQEncoder
from action_recognition_model import ActionRecognitionModel
from loss import ActionLoss, PrivacyLoss
from preprocess import KTHBDQDataset, ConsecutiveTemporalSubsample, MultiScaleCrop, NormalizePixelValues, NormalizeVideo
from privacy_attribute_prediction_model import PrivacyAttributePredictor
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop
import re
import os
from torch.utils.tensorboard import SummaryWriter
import random
from torch import nn
import cv2
import torch
from action_recognition_model import ActionRecognitionModel
# from difference import Difference
# from BDQ import BDQEncoder
from preprocess import KTHBDQDataset

COLAB_PATH = os.getenv('COLAB_PATH')
CHECKPOINT_PATH = "checkpoints" if COLAB_PATH is None else COLAB_PATH

device = "cpu"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ReverseNormalizeVideo(object):
    def __init__(self):
        norm_value = 255
        self.mean = [110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value]
        self.std = [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]

    def __call__(self, x):
        mean_tensor = torch.tensor(self.mean).view(1, x.shape[1], 1, 1)
        std_tensor = torch.tensor(self.std).view(1, x.shape[1], 1, 1)
        return x * std_tensor + mean_tensor

class ReverseNormalizePixelValues(object):
    def __call__(self, x):
        return torch.clamp(x, 0, 1) * 255

def get_sorted_checkpoints(checkpoint_path: str):
    checkpoints = []
    try:
        files = os.listdir(checkpoint_path)
    except FileNotFoundError:
        return checkpoints
    for file in files:
        match = re.search(r'checkpoint_(\d+)\.tar$', file)
        if match:
            checkpoints.append((os.path.join(checkpoint_path, file), int(match.group(1))))
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints

def load_train_checkpoint(E: BDQEncoder, PATH: str):
    checkpoint = torch.load(PATH, weights_only=True, map_location=torch.device(device))
    E.load_state_dict(checkpoint['E_state_dict'])
    E.to(device)

def compute_mean_std(dataloader):
    channel_sum = 0.
    channel_squared_sum = 0.
    num_pixels = 0
    for videos, _1, _2 in dataloader:
        B, T, C, H, W = videos.shape
        videos = videos.view(-1, C, H, W)
        num_pixels += videos.numel() / C
        channel_sum += videos.sum(dim=[0, 2, 3])
        channel_squared_sum += (videos ** 2).sum(dim=[0, 2, 3])
    mean = channel_sum / num_pixels
    std = (channel_squared_sum / num_pixels - mean ** 2).sqrt()
    return mean, std

def main():
    train_transform = Compose([
            ConsecutiveTemporalSubsample(24),
            # MultiScaleCrop(),
            # Resize((224, 224)),
            CenterCrop((224, 224)),
            NormalizePixelValues(),
            NormalizeVideo()
        ])
    reverse_transform = Compose([
        ReverseNormalizeVideo(),
        ReverseNormalizePixelValues(),
    ])
    # basic_transform = Compose([
    #     ConsecutiveTemporalSubsample(24),
    #     CenterCrop((224, 224)),
    #     NormalizePixelValues(),
    # ])
    train_data = KTHBDQDataset(
        root_dir="./datasets/KTH",
        json_path="./datasets/kth_clips.json",
        transform=train_transform,
        # transform=basic_transform,
        split="train",
    )
    train_dataloader = DataLoader(
        train_data,
        batch_size=16,
        num_workers=4,
    )
    # mean, std = compute_mean_std(train_dataloader)
    # print("mean", mean)
    # print("std", std)
    # for input, target_action, target_privacy in train_dataloader:
    #     random_frame = random.randint(0, input.size(1) - 1)
    #     for i in range(input.size(0)):
    #         x = input[i, random_frame, :, :, :]
    #         ToPILImage()(x.clamp(0, 1)).save(f"{i}0.jpg")
    #     break
    bdq = BDQEncoder(hardness=5.0).to(device)
    checkpoints = get_sorted_checkpoints(CHECKPOINT_PATH)
    last_checkpoint_path = checkpoints[-1][0]
    load_train_checkpoint(bdq, last_checkpoint_path)
    for input, target_action, target_privacy in train_dataloader:
        frames = input[0]
        frames = frames.unsqueeze(0)
        b, t, c, h, w = frames.shape
        frames = bdq(frames)
        frames = frames.squeeze(0)
        frames = reverse_transform(frames)
        video = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
        for frame in frames:
            if frame.dtype != torch.uint8:
                frame = (frame).byte()
            frame = frame.permute(1, 2, 0).contiguous().cpu().numpy()
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()
        break

if __name__ == '__main__':
    main()

# img = Image.open("test.jpg").convert("RGB")
# img_tensor = ToTensor()(img).unsqueeze(0)
# output = model(img_tensor)
# ToPILImage()(output.squeeze(0).clamp(0, 1)).save("gaussian_learned.jpg")

# import cv2
# import torch
# from action_recognition_model import ActionRecognitionModel
# from difference import Difference
# from BDQ import BDQEncoder
# from preprocess import KTHBDQDataset
#
# def main():
#     kth = KTHBDQDataset('./KTH', 'kth_clips.json')
#     frames, info = kth[0]
#     t, c, h, w = frames.shape
#     bdq = BDQEncoder()
#     frames = bdq(frames)
#     video = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
#     for frame in frames:
#         if frame.dtype != torch.uint8:
#             frame = (frame.clamp(0, 1) * 255).byte()
#         frame = frame.permute(1, 2, 0).contiguous().cpu().numpy()
#         video.write(frame)
#     cv2.destroyAllWindows()
#     video.release()
#
# if __name__ == '__main__':
#     main()