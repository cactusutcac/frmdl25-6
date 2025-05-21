import os
import json
import random
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from utils import ACTION_LABEL_MAP

class KTHBDQDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None, split=None):
        """
        Args:
            root_dir (str): Path to KTH folder (contains 'boxing', 'handclapping', etc.)
            json_path (str): JSON file with start/end frame info per clip
            transform (callable, optional): Optional transform to be applied.
            split (str, optional): Optional filter for 'train' / 'val' / 'test'
        """
        self.root_dir = root_dir
        self.transform = transform

        with open(json_path, "r") as f:
            all_clips = json.load(f)

        self.data = all_clips if split is None else [clip for clip in all_clips if clip.get("split") == split]
        self.data = self.data[:16]

    def __len__(self):
        return len(self.data)

    def _get_video_path(self, label, video_id):
        """Construct video path using the '_uncomp' suffix."""
        return os.path.join(self.root_dir, label, f"{video_id}_uncomp.avi")


    def _load_clip(self, video_path, start, end):
        """Extract a clip of frames from the video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = max(1, start)
        end = min(end, total_frames)

        indices = torch.arange(start, end+1).long().tolist()
        frames = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = i + 1
            if frame_idx in indices:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tensor = pil_to_tensor(img)
                frames.append(img_tensor)

        cap.release()

        # Apply transformations at the end
        if self.transform:
            return self.transform(torch.stack(frames))
        return torch.stack(frames)  # [T, C, H, W]

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = self._get_video_path(entry["label"], entry["video_id"])
        clip = self._load_clip(video_path, entry["start_frame"], entry["end_frame"])
        return clip, ACTION_LABEL_MAP[entry["label"]], entry["subject"]  # video tensor, action label, and privacy label

class ConsecutiveTemporalSubsample(object):
    """
    Sequentially subsamples num_samples indices from middle of a video formatted
    as a ``torch.Tensor`` of shape (T, C, H, W).
    """

    def __init__(self, num_samples):
        """
        Args:
            num_samples (int): The number of sequential samples to be selected.
        """
        assert isinstance(num_samples, (int))
        self.num_samples = num_samples

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): video tensor with shape (T, C, H, W).
        """
        t = x.shape[0]
        if self.num_samples >= t:
            return x

        offset = (t-self.num_samples) // 2
        return x[offset:(offset+self.num_samples), ...]
    
class MultiScaleCrop(object):
    """
    Randomly chooses a spatial position and scale from a list of scales
    to perform a crop on a video.
    """
    def __init__(self, scales=[1., 1./(2.**(0.25)), 1./(2.**(0.75)), 1./2.]):
        """
        Args:
            scales (list): a list of possible scales for multi-scale cropping.
        """
        self.scales = scales

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): video tensor of shape (T, C, H, W).
        """
        h, w = x.shape[-2:]
        base_size = min(h, w)
        crop_sizes = [int(base_size * scale) for scale in self.scales]

        # Based on the code from https://arxiv.org/abs/2208.02459: choose a random crop width and height
        # from potential ones. The crop size scales can differ by at most 1 index.
        pairs = []
        for i, crop_h in enumerate(crop_sizes):
            for j, crop_w in enumerate(crop_sizes):
                if abs(i-j) <= 1:
                    pairs.append((crop_w, crop_h))

        crop_w, crop_h = random.choice(pairs)

        # Randomly sample the positional offset
        offset_w, offset_h = self._sample_offset(w, h, crop_w, crop_h)

        # Return cropped video
        return x[:, :, offset_h:offset_h+crop_h, offset_w:offset_w+crop_w]

    def _sample_offset(self, w, h, crop_w, crop_h):
        """
        Randomly samples the spatial position offset.
        
        Args:
            w (int): width of video frame.
            h (int): height of video frame.
            crop_w (int): width of cropped frame.
            crop_h (int): height of cropped frame.
        """
        w_step = (w - crop_w) // 4
        h_step = (h - crop_h) // 4

        options = [
            (0, 0),  # top-left
            (4 * w_step, 0),  # top-right
            (0, 4 * h_step),  # bottom-left
            (4 * w_step, 4 * h_step),  # bottom-right
            (2 * w_step, 2 * h_step),  # center

            (0, 2 * h_step),  # center-left
            (4 * w_step, 2 * h_step),  # center-right
            (2 * w_step, 4 * h_step),  # bottom-center
            (2 * w_step, 0),  # top-center
            (1 * w_step, 1 * h_step),  # upper-left quarter
            (3 * w_step, 1 * h_step),  # upper-right quarter
            (1 * w_step, 3 * h_step),  # lower-left quarter
            (3 * w_step, 3 * h_step),  # lower-right quarter
        ]

        return random.choice(options)
