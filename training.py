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
import re
import os

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
            print("delete_old_checkpoints", file)

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

# Avoid randomness to ensure/improve result reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def adverserial_training(train_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor,
                         optimizer_ET: Optimizer, optimizer_P: Optimizer, scheduler_ET: LRScheduler, scheduler_P: LRScheduler,
                         action_loss: ActionLoss, privacy_loss: PrivacyLoss, last_epoch=0, num_epochs=50):
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
        print("save_checkpoint", epoch)
        delete_old_checkpoints()

    print("last_epoch", last_epoch)
    for epoch in tqdm(range(last_epoch, num_epochs)):
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
            loss_action = action_loss.forward(inputs, targets, P)
            loss_action.backward()
            optimizer_ET.step()

            # Freeze E and T, unfreeze and train P
            P.unfreeze()
            E.freeze()
            T.freeze()
            loss_privacy = privacy_loss.forward(inputs, privacies, E)
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
        save_checkpoint(epoch + 1)

        print(f"Epoch {epoch+1}/{num_epochs}, Action Loss: {epoch_loss_action:.4f}, Privacy Loss: {epoch_loss_privacy:.4f}")

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
    print("load_train_checkpoint", PATH)
    # modelA.train()

if __name__ == "__main__":
    global COLAB_PATH
    global CHECKPOINT_PATH
    COLAB_PATH = os.getenv('COLAB_PATH')
    CHECKPOINT_PATH = "checkpoints" if COLAB_PATH is None else COLAB_PATH  # "checkpoints/checkpoint_1.tar"
    print("COLAB_PATH", COLAB_PATH)
    print("CHECKPOINT_PATH", CHECKPOINT_PATH)
    if not os.path.isdir(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
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
    checkpoints = get_sorted_checkpoints()
    last_checkpoint_path = None
    last_epoch = 0
    if len(checkpoints) > 0:
        last_checkpoint_path, last_epoch = checkpoints[-1]
    load_train_checkpoint(E, T, P, optim_ET, optim_P, scheduler_ET, scheduler_P, last_checkpoint_path)
    criterion_action = ActionLoss(encoder=E, target_predictor=T, alpha=1)
    criterion_privacy = PrivacyLoss(privacy_predictor=P)

    adverserial_training(train_dataloader, E, T, P, optim_ET, optim_P, scheduler_ET, scheduler_P, criterion_action, criterion_privacy,
                         last_epoch=last_epoch, num_epochs=num_epochs)