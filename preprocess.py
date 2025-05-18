import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class KTHBDQDataset(Dataset):
    def __init__(self, root_dir, json_path, clip_len=32, resize=(128, 128), split=None):
        """
        Args:
            root_dir (str): Path to KTH folder (contains 'boxing', 'handclapping', etc.)
            json_path (str): JSON file with start/end frame info per clip
            clip_len (int): Number of frames to sample per clip
            resize (tuple): Output frame size
            split (str): Optional filter for 'train' / 'val' / 'test'
        """
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])

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

        indices = torch.linspace(start, end, steps=self.clip_len).long().tolist()
        frames = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = i + 1
            if frame_idx in indices:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = self.transform(img)  # [C, H, W]
                frames.append(img)
                if len(frames) == self.clip_len:
                    break

        cap.release()

        if len(frames) < self.clip_len:
            raise ValueError(f"Clip too short: {video_path} ({start}-{end})")

        return torch.stack(frames)  # [T, C, H, W]

    def __getitem__(self, idx):
        entry = self.data[idx]
        video_path = self._get_video_path(entry["label"], entry["video_id"])
        clip = self._load_clip(video_path, entry["start_frame"], entry["end_frame"])
        return clip  # ready for the Blur module

