import os
import random
from tempfile import TemporaryDirectory
import numpy as np
import torch
from tqdm import tqdm

from BDQ import BDQEncoder
from action_recognition_model import ActionRecognitionModel
from loss import ActionLoss, PrivacyLoss
from preprocess import KTHBDQDataset, ConsecutiveTemporalSubsample, MultiScaleCrop, NormalizePixelValues
from privacy_attribute_prediction_model import PrivacyAttributePredictor
from pytorchvideo.transforms import UniformTemporalSubsample
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize


device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

# Avoid randomness to ensure/improve result reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def adverserial_training(train_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor,
                         optimizer_ET: Optimizer, optimizer_P: Optimizer, scheduler_ET: LRScheduler, scheduler_P: LRScheduler, num_epochs=50):
    for epoch in tqdm(range(num_epochs)):
        # Set all components to training mode
        E.train()
        T.train()
        P.train() #TODO
        epoch_loss_action = 0.0
        epoch_loss_privacy = 0.0
        for inputs, targets, privacies in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            privacies = privacies.to(device)

            # Reset gradients
            optimizer_P.zero_grad()
            optimizer_ET.zero_grad()

            # Freeze P, train E and T together
            P.freeze()
            loss_action = criterion_action.forward(inputs, targets, P)
            loss_action.backward()
            optimizer_ET.step()

            # Freeze E and T, unfreeze and train P
            P.unfreeze()
            E.freeze()
            T.freeze()
            loss_privacy = criterion_privacy.forward(inputs, privacies, E)
            loss_privacy.backward()
            optimizer_P.step()

            # Unfreeze all models, record losses
            E.unfreeze()
            T.unfreeze()

            epoch_loss_action += loss_action
            epoch_loss_privacy += loss_privacy
            
        # Update learning rates
        scheduler_ET.step()
        scheduler_P.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Action Loss: {epoch_loss_action:.4f}, Privacy Loss: {epoch_loss_privacy:.4f}")


if __name__ == "__main__":
    # Set parameters according to https://arxiv.org/abs/2208.02459
    num_epochs = 50
    lr = 0.001
    batch_size = 16
    consecutive_frames = 32
    output_size = (224, 224)

    # Load KTH dataset. Apply transformation sequence according to Section 4.2 in https://arxiv.org/abs/2208.02459
    transform = Compose([
        ConsecutiveTemporalSubsample(32), # first, sample 32 consecutive frames
        MultiScaleCrop(), # then, apply randomized multi-scale crop
        Resize((224, 224)), # then, resize to (224, 224)
        NormalizePixelValues(),
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
        num_workers=4
    )
    # Initialize the BDQEncoder (E), the action attribute predictor (T),
    # and the privacy attribute predictor (P)
    E = BDQEncoder(hardness=5.0)
    T = ActionRecognitionModel(fine_tune=True, num_classes=6)
    P = PrivacyAttributePredictor(num_privacy_classes=25)

    # Initialize optimizer, scheduler and loss functions
    optim_ET = SGD(params=list(E.parameters())+list(T.parameters()), lr=lr)
    optim_P = SGD(params=list(P.parameters()), lr=lr)
    scheduler_ET = CosineAnnealingLR(optimizer=optim_ET, T_max=num_epochs)
    scheduler_P = CosineAnnealingLR(optimizer=optim_P, T_max=num_epochs)
    criterion_action = ActionLoss(encoder=E, target_predictor=T, alpha=1)
    criterion_privacy = PrivacyLoss(privacy_predictor=P)

    adverserial_training(train_dataloader, E, T, P, optim_ET, optim_P, scheduler_ET, scheduler_P, num_epochs=num_epochs)