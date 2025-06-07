KTH_DATA_DIR = '/kaggle/input/kth-dataset-copy/KTH/KTH'
KTH_LABELS_DIR = '/kaggle/input/kth-dataset-copy/kth_clips.json'

import re
import os
import cv2
import json
import random

from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.transforms.functional import pil_to_tensor
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize, Resize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
print(device)

os.environ['COLAB_PATH'] = '/kaggle/working/checkpoints'

# ## 3. Define Classes

# ### 3.1 KTH Dataset and Transformation Classes

ACTION_LABEL_MAP = {
    "boxing": 0,
    "handclapping": 1,
    "handwaving": 2,
    "jogging": 3,
    "running": 4,
    "walking": 5,
}

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

# ### 3.2 Loss Functions

class ActionLoss(nn.Module):
    """
    Args:
        encoder: the BDQ encoder
        target_predictor: 3D CNN N for predicting target action attribute
        alpha: the adversarial weight for trade-off between action and privacy recognition
    """
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def entropy(self, x, dim=1, eps=1e-6):
        x = torch.clamp(x, eps, 1)
        return -torch.mean(torch.sum(x * torch.log(x), dim=dim))

    """
    Args:
        T_pred: predicted target labels for the input video
        P_pred: predicted privacy labels for the input video
        L_action: the ground-truct action labels of the inputs
    """
    def forward(self, T_pred, P_pred, L_action):
        loss = self.cross_entropy(T_pred, L_action) - self.alpha * self.entropy(P_pred)
        return loss

class PrivacyLoss(nn.Module):
    """
    Args:
        privacy_predictor: 2D CNN for predicting the privacy attribute
    """
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    """
    Args:
        P_pred: predicted privacy labels for the input video
        L_privacy: the ground-truth privacy labels
        fixed_encoder: the (fixed) BDQ encoder
    """
    def forward(self, P_pred, L_privacy):
        loss = self.cross_entropy(P_pred, L_privacy)
        return loss


# ### 3.3 BDQ Encoder Modules

class LearnableGaussian(nn.Module):
    def __init__(self, kernel_size=5, init_sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.tensor(init_sigma))

    def forward(self, x):
        # Make sure sigma is positive
        sigma = self.sigma

        # x: (B, T, C, H, W)
        # merge B, T and C to apply/learn same kernel for all channels
        B, T, C, H, W = x.shape
        x = x.view(-1, 1, H, W)
        C_kernel = 1 #TODO initially was =C

        # Create 1D kernel
        k = self.kernel_size // 2
        coords = torch.arange(-k, k + 1, dtype=torch.float32, device=x.device)
        gauss = torch.exp(-0.5 * (coords / sigma)**2)
        gauss = gauss / gauss.sum()

        # Make 2D kernel
        kernel = 0.5 / (torch.pi * (sigma ** 2)) * torch.outer(gauss, gauss)
        kernel = kernel.expand(C_kernel, 1, self.kernel_size, self.kernel_size)

        # Apply depthwise convolution
        x = F.conv2d(x, kernel, padding=k, groups=C_kernel)
        return x.view(B, T, C, H, W)

class Difference(nn.Module):
    def __init__(self):
        super(Difference, self).__init__()
        self.bvj = None

    """
    Args:
        x: the input frames tensor of shape (B, T, C, H, W), i.e. video with T frames
    """
    def forward(self, x):
        d = x.roll(-1, dims=1) - x
        return d

class DifferentiableQuantization(nn.Module):
    def __init__(self, num_bins=15, hardness=5.0, normalize_input=True, rescale_output=True):
        """
        Args:
            num_bins (int): Number of quantization bins N = 2^k (default 15).
            hardness (float): Controls sigmoid sharpness; higher = closer to step function.
            normalize_input (bool): Whether to normalize input to [0, num_bins] before quantizing.
            rescale_output (bool): Whether to rescale output back to input's original value range.
        """
        super().__init__()
        self.num_bins = num_bins
        self.hardness = hardness
        self.normalize_input = normalize_input
        self.rescale_output = rescale_output

        # Initialize bin centers at [0.5, 1.5, ..., 14.5] for num_bins = 15
        init_bins = torch.linspace(0.5, num_bins - 0.5, steps=num_bins)
        self.bins = nn.Parameter(init_bins)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W)
        Returns:
            Tensor: Quantized output of shape (B, T, C, H, W)
        """
        orig_min, orig_max = x.min(), x.max() #TODO is it batch min/max?

        if self.normalize_input:
            qmin = 0.0
            qmax = float(self.num_bins)
            scale = (orig_max - orig_min) / (qmax - qmin)
            scale = max(scale, 1e-4)
            x = (x - orig_min) / (orig_max - orig_min + 1e-4) * (qmax - qmin)

        # Expand for broadcasting
        x_expanded = x.unsqueeze(-1)                        # Shape: [B, T, C, H, W, 1]
        bin_centers = self.bins.view(1, 1, 1, 1, 1, -1).to(device)        # Shape: [1, 1, 1, 1, 1, num_bins]

        # Sum of sigmoid activations
        y = torch.sigmoid(self.hardness * (x_expanded - bin_centers)).sum(dim=-1)

        if self.normalize_input and self.rescale_output:
            y = y * scale + orig_min

        return y


# ### 3.4 BDQ Encoder and Label Predictors

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

    # def freeze(self):
    #     """
    #     Freezes the parameters to prevent/pause learning.
    #     """
    #     for param in self.parameters():
    #         param.requires_grad = False

    # def unfreeze(self):
    #     """
    #     Resumes learning for BDQ encoder parameters.
    #     """
    #     for param in self.parameters():
    #         param.requires_grad = True

class ActionRecognitionModel(nn.Module):
    def __init__(self, fine_tune, num_classes = 400, id_to_classname = None):
        super(ActionRecognitionModel, self).__init__()

        # From action recognition file global variables
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 8
        sampling_rate = 8
        frames_per_second = 30
        clip_duration = (num_frames * sampling_rate) / frames_per_second
        self.start_sec = 0
        self.end_sec = self.start_sec + clip_duration

        model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
        model = model.eval()
        model = model.to(device)
        if fine_tune:
            for param in model.parameters():
                param.requires_grad = False
            model.blocks[-1].proj = nn.Linear(in_features = model.blocks[-1].proj.in_features, out_features = num_classes)
            for param in model.blocks[-1].proj.parameters():
                param.requires_grad = True
        self.model = model
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose([UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    ShortSideScale(size = side_size),
                    CenterCrop((crop_size, crop_size))])
        )
        self.id_to_classname = id_to_classname

    #accepts [B?, C=3, T=num_frames?, crop_size, crop_size]
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input video (batched) of shape (B, T, C, H, W).

        outputs:
            y (string): predicted action label.
        """
        # If not batched, make sample of 1 batch
        if len(x.shape) == 4:
            x = x.unsqueeze(dim=0)

        # Transpose channel and temporal dimension
        x = torch.transpose(x, -3, -4)
        logits = self.model(x)  # Get prediction logits from 3d resnet. Shape: (B, num_classes)

        # Apply softmax to get and return probabilities of each label
        logits_softmax = F.softmax(logits, dim=1)

        return logits_softmax

    def test(self, video_path):
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec = self.start_sec, end_sec =self.end_sec)
        video_data = self.transform(video_data)
        inputs = video_data["video"]
        return self.predict(inputs)

    def predict(self, inputs):
        inputs = inputs.to(device)
        preds = self.forward(inputs[None, ...])
        post_act = torch.nn.Softmax(dim = 1)
        preds = post_act(preds)
        pred_classes = preds.topk(k = 1).indices[0]
        pred_class_names = [self.id_to_classname[int(i)] for i in pred_classes]
        return pred_class_names

    # def freeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = False

    # def unfreeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = True
# Adapted from: https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/

class PrivacyAttributePredictor(nn.Module):
    """
    Privacy Attribute Prediction Model.
    Uses a 2D ResNet-50 to predict privacy attributes from BDQ-encoded video frames.
    The softmax outputs from each frame are averaged.
    """
    def __init__(self, num_privacy_classes, pretrained_resnet=True):
        """
        Args:
            num_privacy_classes (int): The number of privacy attribute classes to predict.
            pretrained_resnet (bool): Whether to use ImageNet pre-trained weights for ResNet-50.
        """
        super().__init__()
        self.num_privacy_classes = num_privacy_classes

        # Load a 2D ResNet-50 model
        if pretrained_resnet:
            self.resnet_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet_feature_extractor = models.resnet50(weights=None)
        for param in self.resnet_feature_extractor.parameters():
            param.requires_grad = False
        # Replace the final fully connected layer for the new number of privacy classes
        num_ftrs = self.resnet_feature_extractor.fc.in_features
        self.resnet_feature_extractor.fc = nn.Linear(num_ftrs, num_privacy_classes)
        for param in self.resnet_feature_extractor.fc.parameters():
            param.requires_grad = True

    def forward(self, bdq_encoded_video):
        """
        Forward pass for the privacy attribute predictor.

        Args:
            bdq_encoded_video (torch.Tensor): The output from the BDQ encoder.
                Shape: (B, T, C, H, W), where
                B = batch size
                T = number of time steps/frames
                C = number of channels
                H = height
                W = width

        Returns:
            torch.Tensor: Averaged softmax probabilities for privacy attributes.
                          Shape: (B, num_privacy_classes)
        """
        B, T, C, H, W = bdq_encoded_video.shape

        # ResNet50 expects input of shape (N, C, H, W).
        # We need to process each of the T frames for each video in the batch.
        # Reshape to (B*T, C, H, W) to pass all frames through ResNet in one go.
        video_reshaped_for_resnet = bdq_encoded_video.contiguous().view(B * T, C, H, W)

        # Get logits from the ResNet feature extractor for all (B*T) frames
        logits_all_frames = self.resnet_feature_extractor(video_reshaped_for_resnet) # Shape: (B*T, num_privacy_classes)

        # Apply softmax to get probabilities for each frame
        softmax_all_frames = F.softmax(logits_all_frames, dim=1) # Shape: (B*T, num_privacy_classes)

        # Reshape back to (B, T, num_privacy_classes) to separate frames per video
        softmax_per_frame_per_video = softmax_all_frames.view(B, T, self.num_privacy_classes)

        # Average the softmax outputs over the T frames for each video in the batch
        # as described in the paper (Section 4.2 Validation & Section 4.3 Results explanation).
        averaged_softmax_predictions = torch.mean(softmax_per_frame_per_video, dim=1) # Shape: (B, num_privacy_classes)

        return averaged_softmax_predictions

    # def freeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = False

    # def unfreeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = True


# ## 4. Define Model Training Function

# Setup checkpointing
COLAB_PATH = os.getenv('COLAB_PATH')
CHECKPOINT_PATH = "checkpoints" if COLAB_PATH is None else COLAB_PATH  # "checkpoints/checkpoint_1.tar"
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)


def get_sorted_checkpoints():
    checkpoints = []
    try:
        files = os.listdir(CHECKPOINT_PATH)
    except FileNotFoundError:
        return checkpoints
    for file in files:
        match = re.search(r'checkpoint_(\d+)\.tar$', file)
        if match:
            checkpoints.append((os.path.join(CHECKPOINT_PATH, file), int(match.group(1))))
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints

def delete_old_checkpoints():
    checkpoints = get_sorted_checkpoints()
    if len(checkpoints) > 2:
        for file, _ in checkpoints[:-2]:
            os.remove(file)

def compute_accuracy(input, target_action, target_privacy):
    """
    Computes action and privacy prediction accuracy
    Args:
        input: the input (batched) video tensor
        target_action: target labels for action attribute
        target_privacy: target labels for privacy attribute
    """
    with torch.no_grad():
        input_encoded = E.forward(input)
        T_pred = T.forward(input_encoded).argmax(dim=1)
        P_pred = P.forward(input_encoded).argmax(dim=1)

        action_acc = torch.sum(T_pred == target_action)
        privacy_acc = torch.sum(P_pred == target_privacy)

        return action_acc, privacy_acc

def train_once(train_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor,
               action_loss: ActionLoss, privacy_loss: PrivacyLoss, optimizer_ET: Optimizer, optimizer_P: Optimizer):
    """
    Function to perform one training epoch of adverserial training from https://arxiv.org/abs/2208.02459
    Args:
        train_dataloader: DataLoader for the training split of the KTH dataset
        E: the BDQ encoder
        T: 3d resnet50 for predicting target action attributes
        P: 2d resnet50 for predicting target privacy attributes
        action_loss: criterion for optimizing action attribute prediction
        privacy_loss: criterion for optimizing privacy attribute prediction
        optimizer_ET: SGD optimizer for the encoder and action attribute predictor
        optimizer_P: SGD optimizer for the privacy attribute predictor
    """
    # Set all components to training mode
    E.train()
    T.train()
    P.train()

    total_loss_action = torch.tensor(0.)
    total_loss_privacy = torch.tensor(0.)
    total_acc_action = torch.tensor(0.)
    total_acc_privacy = torch.tensor(0.)

    for input, target_action, target_privacy in tqdm(train_dataloader, total=len(train_dataloader), desc="Training epoch...", unit="batch", position=1, leave=False):
        input = input.to(device)
        target_action = target_action.to(device)
        target_privacy = target_privacy.to(device)

        # Reset gradients
        # optimizer_P.zero_grad()
        optimizer_ET.zero_grad()

        # Freeze P, train E and T together
        # P.freeze()
        input_encoded = E.forward(input)
        action_pred = T.forward(input_encoded)
        frozen_privacy_pred = P.forward(input_encoded)
        loss_action = action_loss.forward(action_pred, frozen_privacy_pred, target_action)
        loss_action.backward()
        optimizer_ET.step()

        optimizer_P.zero_grad()

        # Freeze E and T, unfreeze and train P
        # P.unfreeze()
        # E.freeze()
        # T.freeze()
        frozen_input_encoded = E.forward(input)
        privacy_pred = P.forward(frozen_input_encoded)
        loss_privacy = privacy_loss.forward(privacy_pred, target_privacy)
        loss_privacy.backward()
        optimizer_P.step()

        # Unfreeze all models, record losses
        # E.unfreeze()
        # T.unfreeze()

        # Compute statistics
        acc_action, acc_privacy = compute_accuracy(input, target_action, target_privacy)

        total_loss_action += loss_action.item()
        total_loss_privacy += loss_privacy.item()

        total_acc_action += acc_action.item()
        total_acc_privacy += acc_privacy.item()

    # Average out accuracies
    total_acc_action /= len(train_dataloader.dataset)
    total_acc_privacy /= len(train_dataloader.dataset)

    return total_loss_action, total_loss_privacy, total_acc_action, total_acc_privacy

def validate_once(val_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor,
                  action_loss: ActionLoss, privacy_loss: PrivacyLoss):
    """
    Function to perform one validation epoch of adverserial training from https://arxiv.org/abs/2208.02459
    Args:
        val_dataloader: DataLoader for the validation split of the KTH dataset
        E: the BDQ encoder
        T: 3d resnet50 for predicting target action attributes
        P: 2d resnet50 for predicting target privacy attributes
        action_loss: criterion for optimizing action attribute prediction
        privacy_loss: criterion for optimizing privacy attribute prediction
    """
    E.eval()
    T.eval()
    P.eval()

    with torch.no_grad():

        total_loss_action = torch.tensor(0.)
        total_loss_privacy = torch.tensor(0.)
        total_acc_action = torch.tensor(0.)
        total_acc_privacy = torch.tensor(0.)

        for input, target_action, target_privacy in tqdm(val_dataloader, total=len(val_dataloader), desc="Validating epoch...", unit="batch", position=1, leave=False):
            input = input.to(device)
            target_action = target_action.to(device)
            target_privacy = target_privacy.to(device)

            # Perform evaluation with models on respective inputs
            input_encoded = E.forward(input)
            action_pred = T.forward(input_encoded)
            privacy_pred = P.forward(input_encoded)

            # Compute statistics
            loss_action = action_loss.forward(action_pred, privacy_pred, target_action)
            loss_privacy = privacy_loss.forward(privacy_pred, target_privacy)

            acc_action, acc_privacy = compute_accuracy(input, target_action, target_privacy)

            total_loss_action += loss_action.item()
            total_loss_privacy += loss_privacy.item()
            total_acc_action += acc_action.item()
            total_acc_privacy += acc_privacy.item()

        # Average out accuracies
        total_acc_action /= len(val_dataloader.dataset)
        total_acc_privacy /= len(val_dataloader.dataset)

        return total_loss_action, total_loss_privacy, total_acc_action, total_acc_privacy

def adverserial_training(train_dataloader: DataLoader, val_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel,
                         P: PrivacyAttributePredictor, optimizer_ET: Optimizer, optimizer_P: Optimizer, scheduler_ET: LRScheduler,
                         scheduler_P: LRScheduler, action_loss: ActionLoss, privacy_loss: PrivacyLoss, last_epoch=0, num_epochs=50):
    """
    Function encapsulating the whole adverserial training process from https://arxiv.org/abs/2208.02459
    Args:
        train_dataloader: DataLoader for the training split of the KTH dataset
        val_dataloader: DataLoader for the validation split of the KTH dataset
        E: the BDQ encoder
        T: 3d resnet50 for predicting target action attributes
        P: 2d resnet50 for predicting target privacy attributes
        optimizer_ET: SGD optimizer for the encoder and action attribute predictor
        optimizer_P: SGD optimizer for the privacy attribute predictor
        scheduler_ET: learning rate scheduler for updating learning rate each epoch for optimizer_ET
        scheduler_P: learning rate scheduler for updating learning rate each epoch for optimizer_P
        action_loss: criterion for optimizing action attribute prediction
        privacy_loss: criterion for optimizing privacy attribute prediction
        last_epoch (optional, int): checkpoint of last saved epoch
        num_epochs (optional, int): number of epochs to train for (default=50)
    """
    def save_checkpoint(epoch: int):
        torch.save({
            'E_state_dict': E.state_dict(),
            'T_state_dict': T.state_dict(),
            'P_state_dict': P.state_dict(),
            'optim_ET_state_dict': optimizer_ET.state_dict(),
            'optim_P_state_dict': optimizer_P.state_dict(),
            'scheduler_ET_state_dict': scheduler_ET.state_dict(),
            'scheduler_P_state_dict': scheduler_P.state_dict(),
        }, os.path.join(CHECKPOINT_PATH, f"checkpoint_{epoch}.tar"))
        delete_old_checkpoints()

    action_accuracies_adv_train = []
    privacy_accuracies_adv_train = []

    with tqdm(range(last_epoch, num_epochs), total=num_epochs, initial=last_epoch, desc="Averserial training", unit="epoch", position=0, leave=True) as progress_loader:
        for epoch in progress_loader:
            train_loss_action, train_loss_privacy, train_acc_action, train_acc_privacy = train_once(train_dataloader=train_dataloader, E=E, T=T, P=P,
                                                                                                    action_loss=action_loss, privacy_loss=privacy_loss,
                                                                                                    optimizer_ET=optimizer_ET, optimizer_P=optimizer_P)

            val_loss_action, val_loss_privacy, val_acc_action, val_acc_privacy = validate_once(val_dataloader=val_dataloader, E=E, T=T, P=P,
                                                                                            action_loss=action_loss, privacy_loss=privacy_loss)

            # Update learning rates
            scheduler_ET.step()
            scheduler_P.step()
            save_checkpoint(epoch + 1)

            # Display statistics
            progress_loader.set_postfix(action_loss=val_loss_action.numpy(), privacy_loss=val_loss_privacy.numpy(),
                                         action_accuracy=val_acc_action.numpy(), privacy_accuracy= val_acc_privacy.numpy())
            action_accuracies_adv_train.append(val_acc_action.item())
            privacy_accuracies_adv_train.append(val_acc_privacy.item())
            progress_loader.refresh()
            # print(f"Epoch {epoch+1}/{num_epochs}, Action Loss: {val_loss_action:.4f}, Privacy Loss: {val_loss_privacy:.4f}")
            # print(f"Action accuracy: {val_acc_action:.4f}, Privacy accuracy: {val_acc_privacy:.4f}")
    
    # Free resources before plotting/logging
    del train_dataloader
    del val_dataloader
    import gc
    gc.collect()

    return action_accuracies_adv_train, privacy_accuracies_adv_train

def load_train_checkpoint(E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor,
               optim_ET: Optimizer, optim_P: Optimizer, scheduler_ET: LRScheduler, scheduler_P: LRScheduler, PATH: str | None):
    if PATH is None:
        return
    checkpoint = torch.load(PATH, weights_only=True)
    E.load_state_dict(checkpoint['E_state_dict'])
    E.to(device)
    T.load_state_dict(checkpoint['T_state_dict'])
    T.to(device)
    P.load_state_dict(checkpoint['P_state_dict'])
    P.to(device)
    optim_ET.load_state_dict(checkpoint['optim_ET_state_dict'])
    optim_P.load_state_dict(checkpoint['optim_P_state_dict'])
    scheduler_ET.load_state_dict(checkpoint['scheduler_ET_state_dict'])
    scheduler_P.load_state_dict(checkpoint['scheduler_P_state_dict'])


# ## 5. Adversarial Training 


# Set parameters according to https://arxiv.org/abs/2208.02459
num_epochs = 50
lr = 0.001
batch_size = 4
consecutive_frames = 24 # FixMe
crop_size = (224, 224)

# Load KTH dataset. Apply transformation sequence according to Section 4.2 in https://arxiv.org/abs/2208.02459
train_transform = Compose([
    ConsecutiveTemporalSubsample(consecutive_frames), # first, sample 32 consecutive frames
    MultiScaleCrop(), # then, apply randomized multi-scale crop
    Resize(crop_size), # then, resize to (224, 224)
    NormalizePixelValues(), # (also normalize pixel values for pytorch)
])
train_data = KTHBDQDataset(
    root_dir=KTH_DATA_DIR,
    json_path=KTH_LABELS_DIR,
    transform=train_transform,
    split="train",
)
train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=4,
)
# Load validation dataset according to the same Section 4.2
val_transform = Compose([
    ConsecutiveTemporalSubsample(consecutive_frames), # first sample 32 consecutive frames
    CenterCrop(crop_size),  # then, we apply a center crop of (224, 224) without scaling (resizing)
    NormalizePixelValues(), # (also normalize pixel values for pytorch)
])
val_data = KTHBDQDataset(
    root_dir=KTH_DATA_DIR,
    json_path=KTH_LABELS_DIR,
    transform=val_transform,
    split="val",
)
val_dataloader = DataLoader(
    val_data,
    batch_size=batch_size,
    num_workers=4,
)
# Test set for "Validation" in Section 4.2 
test_transform = Compose([
    ConsecutiveTemporalSubsample(consecutive_frames), # first sample 32 consecutive frames
    CenterCrop(crop_size),  # then, we apply a center crop of (224, 224) without scaling (resizing)
    NormalizePixelValues(), # (also normalize pixel values for pytorch)
])
test_data = KTHBDQDataset(
    root_dir=KTH_DATA_DIR,
    json_path=KTH_LABELS_DIR,
    transform=test_transform,
    split="test",
)
test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    num_workers=4,
)

# Initialize the BDQEncoder (E), the action attribute predictor (T),
# and the privacy attribute predictor (P)
E = BDQEncoder(hardness=5.0).to(device)
T = ActionRecognitionModel(fine_tune=True, num_classes=6).to(device)
P = PrivacyAttributePredictor(num_privacy_classes=25).to(device)

# Initialize optimizer, scheduler and loss functions
optim_ET = SGD(params=list(E.parameters())+list(T.parameters()), lr=lr)
optim_P = SGD(params=P.parameters(), lr=lr)
scheduler_ET = CosineAnnealingLR(optimizer=optim_ET, T_max=num_epochs)
scheduler_P = CosineAnnealingLR(optimizer=optim_P, T_max=num_epochs)
checkpoints = get_sorted_checkpoints()
last_checkpoint_path = None
last_epoch = 0
if len(checkpoints) > 0:
    last_checkpoint_path, last_epoch = checkpoints[-1]
load_train_checkpoint(E, T, P, optim_ET, optim_P, scheduler_ET, scheduler_P, last_checkpoint_path)
criterion_action = ActionLoss(alpha=1)
criterion_privacy = PrivacyLoss()

action_accuracies_adv_train, privacy_accuracies_adv_train = adverserial_training(train_dataloader=train_dataloader, val_dataloader=val_dataloader, E=E, T=T, P=P,
                      optimizer_ET=optim_ET, optimizer_P=optim_P, scheduler_ET=scheduler_ET,
                      scheduler_P=scheduler_P, action_loss=criterion_action, privacy_loss=criterion_privacy,
                      last_epoch=last_epoch, num_epochs=num_epochs)

# ## 6. Validation 

from torchvision.models.video import r3d_18
from torchvision.models import resnet50

def validate_frozen_bdq(E, train_dataloader, test_dataloader, device):
    # Freeze BDQ encoder
    E.eval()
    for param in E.parameters():
        param.requires_grad = False

    # Action model: 3D ResNet-18 (ResNet-50 if you switch to it)
    model_action = r3d_18(pretrained=True)
    model_action.fc = nn.Linear(model_action.fc.in_features, 6)
    model_action = model_action.to(device)

    # Privacy model: 2D ResNet-50
    model_privacy = resnet50(pretrained=True)
    model_privacy.fc = nn.Linear(model_privacy.fc.in_features, 25)
    model_privacy = model_privacy.to(device)

    # Optimizers
    opt_action = torch.optim.SGD(model_action.parameters(), lr=1e-3, momentum=0.9)
    opt_privacy = torch.optim.SGD(model_privacy.parameters(), lr=1e-3, momentum=0.9)

    # Training loop
    for epoch in range(50):
        model_action.train()
        model_privacy.train()
        total_action_loss = 0.0
        total_privacy_loss = 0.0

        for batch in train_dataloader:
            x, y_action, y_privacy = batch["video"].to(device), batch["action"].to(device), batch["identity"].to(device)

            with torch.no_grad():
                feat = E(x)

            # Action model
            out_action = model_action(feat)
            loss_action = F.cross_entropy(out_action, y_action)
            opt_action.zero_grad()
            loss_action.backward()
            opt_action.step()

            # Privacy model
            feat_avg = feat.mean(dim=2)
            out_privacy = model_privacy(feat_avg)
            loss_privacy = F.cross_entropy(out_privacy, y_privacy)
            opt_privacy.zero_grad()
            loss_privacy.backward()
            opt_privacy.step()

            total_action_loss += loss_action.item()
            total_privacy_loss += loss_privacy.item()

        print(f"[Epoch {epoch+1}] Action Loss: {total_action_loss:.3f} | Privacy Loss: {total_privacy_loss:.3f}")

    # Evaluation
    model_action.eval()
    model_privacy.eval()
    correct_action = 0
    correct_privacy = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            x, y_action, y_privacy = batch["video"].to(device), batch["action"].to(device), batch["identity"].to(device)

            feat = E(x)
            out_action = model_action(feat)
            out_privacy = model_privacy(feat.mean(dim=2))

            correct_action += (out_action.argmax(dim=1) == y_action).sum().item()
            correct_privacy += (out_privacy.argmax(dim=1) == y_privacy).sum().item()
            total += x.size(0)

    print("\n[Validation Results with frozen BDQ encoder]")
    print(f"Action Accuracy:  {correct_action / total * 100:.2f}%")
    print(f"Privacy Accuracy: {correct_privacy / total * 100:.2f}%")

validate_frozen_bdq(E, train_dataloader, test_dataloader, device)

# ## 7. Logging and Visualization 

def save_quantizer_mapping(dq_module, output_csv_path="quant_steps.csv", device="cpu"):
    # Parameters
    num_bins = dq_module.num_bins
    hardness = dq_module.hardness
    normalize = dq_module.normalize_input
    rescale = dq_module.rescale_output

    # Input values in normalized space: [0, num_bins]
    x_vals = torch.linspace(0, num_bins, 1000, device=device).view(-1, 1)

    # Initial bin centers [0.5, 1.5, ..., 14.5]
    init_bins = torch.linspace(0.5, num_bins - 0.5, steps=num_bins).to(device).view(1, -1)

    # Learned bins
    learned_bins = dq_module.bins.detach().to(device).view(1, -1)

    # Quantization function: sum of sigmoids
    def quant_output(x, bins):
        return torch.sigmoid(hardness * (x - bins)).sum(dim=-1)

    # Evaluate
    with torch.no_grad():
        y_init = quant_output(x_vals, init_bins)
        y_learned = quant_output(x_vals, learned_bins)

        # Optional: rescale output like your quantizer does
        if rescale:
            y_init = y_init * (1.0) + 0.0  # No orig_min/max: we stay in normalized space
            y_learned = y_learned * (1.0) + 0.0

        # Convert to numpy
        x_vals_np = x_vals.squeeze().cpu().numpy()
        y_init_np = y_init.squeeze().cpu().numpy()
        y_learned_np = y_learned.squeeze().cpu().numpy()

    # Save as CSV
    df = pd.DataFrame({
        "input": x_vals_np,
        "init_output": y_init_np,
        "learned_output": y_learned_np
    })
    df.to_csv(output_csv_path, index=False)

save_quantizer_mapping(E.encoder[2], "quant_steps_kth_adv_train.csv", device="cuda")

epochs = list(range(last_epoch + 1, num_epochs + 1))
log_df = pd.DataFrame({
    'epoch': epochs,
    'action_accuracy': action_accuracies_adv_train,
    'privacy_accuracy': privacy_accuracies_adv_train
})

log_df['action_accuracy'] *= 100
log_df['privacy_accuracy'] *= 100

log_df.to_csv('accuracy_log_kth_adv_train.csv', index=False)

plt.figure()
plt.plot(log_df['epoch'], log_df['action_accuracy'], label='Action Accuracy')
plt.plot(log_df['epoch'], log_df['privacy_accuracy'], label='Privacy Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy (%) vs. Epoch')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot_kth_adv_train.png')
plt.close()
