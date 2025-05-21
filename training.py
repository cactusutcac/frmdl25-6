import random
import numpy as np
import torch
from tqdm import tqdm

from BDQ import BDQEncoder
from action_recognition_model import ActionRecognitionModel
from loss import ActionLoss, PrivacyLoss
from preprocess import KTHBDQDataset, ConsecutiveTemporalSubsample, MultiScaleCrop
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms


# Avoid randomness to ensure/improve result reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.use_deterministic_algorithms(True)

# Set parameters according to https://arxiv.org/abs/2208.02459
epochs = 50
lr = 0.001
batch_size = 16
consecutive_frames = 32
output_size = (224, 224)

device = torch.accelerator.current_accelerator if torch.accelerator.is_available() else "cpu"

# Load KTH dataset. Apply transformation sequence according to Section 4.2 in https://arxiv.org/abs/2208.02459
transform = transforms.Compose([
    ConsecutiveTemporalSubsample(32), # first, sample 32 consecutive frames
    MultiScaleCrop(), # then, apply randomized multi-scale crop
    transforms.Resize((224, 224)), # then, resize to (224, 224)
    transforms.Lambda(lambda x: x / 255.)
])
train_data = KTHBDQDataset(
    root_dir="./KTH",
    json_path="kth_clips.json",
    transform=transform,
    split="train",
)
train_dataloader = DataLoader(
    train_data, 
    batch_size=batch_size,
    num_workers=8
)

# Initialize the BDQEncoder (E), the action attribute predictor (T),
# and the privacy attribute predictor (P)
E = BDQEncoder()
T = ActionRecognitionModel(fine_tune=True, num_classes=6)
P = None

# Initialize optimizer, scheduler and loss functions
optim_E = SGD(params=E.parameters())
optim_PT = SGD(params=list(T.parameters())+list(), lr=lr) # TODO: add privacy predictor parameters
scheduler_E = CosineAnnealingLR(optimizer=optim_E, T_max=epochs)
scheduler_PT = CosineAnnealingLR(optimizer=optim_PT, T_max=epochs)
criterion_action = ActionLoss(encoder=E, target_predictor=T, alpha=1)
criterion_privacy = PrivacyLoss(privacy_predictor=P)

for epoch in tqdm(range(epochs)):
    # Set all components to training mode
    E.train()
    T.train()
    # P.train()
    epoch_loss_action = 0.0
    epoch_loss_privacy = 0.0
    for inputs, targets, privacies in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        privacies = privacies.to(device)

        # Reset gradients
        optim_PT.zero_grad()
        optim_E.zero_grad()

        # Freeze P, train E and T together
        # P.freeze()
        loss_action = criterion_action(inputs, targets, P)
        loss_action.backward()
        optim_PT.step()

        # Freeze E and T, unfreeze and train P
        # P.unfreeze()
        E.freeze()
        T.freeze()
        loss_privacy = criterion_privacy(inputs, privacies, E)
        loss_privacy.backward()
        optim_E.step()

        # Unfreeze all models, record losses
        E.unfreeze()
        T.unfreeze()

        epoch_loss_action += loss_action
        epoch_loss_privacy += loss_privacy
        
    # Update learning rates
    scheduler_E.step()
    scheduler_PT.step()

    print(f"Epoch {epoch+1}/{epochs}, Action Loss: {epoch_loss_action:.4f}, Privacy Loss: {epoch_loss_privacy:.4f}")