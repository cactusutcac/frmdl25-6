import os
import json
import random
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from datasets.utils_kth import ACTION_LABEL_MAP

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

class IXMASBDQDataset(Dataset):
    def __init__(self, root_dir, json_path, transform=None, split=None):
        """
        Args:
            root_dir (str): Folder containing .avi video files.
            json_path (str): Path to ixmas_clips.json
            transform (callable, optional): Optional transform to apply to [T, C, H, W] tensor.
            split (str): 'train', 'val', or 'test'; filters dataset.
        """
        self.root_dir = root_dir
        self.transform = transform

        with open(IXMAS_LABELS_DIR, "r") as f:
            all_clips = json.load(f)
        
        self.data = [c for c in all_clips if c["split"] == split] if split else all_clips
        self.action_label_map = {label: i for i, label in enumerate(sorted(set(c["label"] for c in self.data)))}

    def __len__(self):
        return len(self.data)

    def _load_clip(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tensor = pil_to_tensor(img)
            frames.append(img_tensor)

        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames extracted from {video_path}")

        clip = torch.stack(frames)  # [T, C, H, W]
        return self.transform(clip) if self.transform else clip

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = os.path.join(self.root_dir, entry["video_id"])
        clip = self._load_clip(video_path)
        action_label = self.action_label_map[entry["label"]]
        subject_id = entry["subject"]
        return clip, action_label, subject_id

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

class NormalizePixelValues(object):
    """
    Normalizes pixel values to be in the range [0., 1.] instead of the hex format.
    """
    def __init__(self, eps=1e-6):
        """
        Args:
            eps (float): small offset to prevent edge values.
        """
        self.eps = eps

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): an image-like tensor whose values to normalize.
        """
        return torch.clamp(x / 255., self.eps, 1.-self.eps)

class NormalizeVideo(object):
    def __init__(self):
        norm_value = 255
        self.mean = [110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value] # reused from https://github.com/suakaw/BDQ_PrivacyAR/blob/main/action-recognition-pytorch-entropy/train.py#L247
        self.std = [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]

    def __call__(self, x): # [T, C, H, W]
        mean_tensor = torch.tensor(self.mean).view(1, x.shape[1], 1, 1)
        std_tensor = torch.tensor(self.std).view(1, x.shape[1], 1, 1)
        return (x - mean_tensor) / std_tensor
