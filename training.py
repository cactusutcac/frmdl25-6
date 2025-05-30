import os
import random
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
from torchvision.transforms import Compose, Resize, CenterCrop
import re
import os

# Setup checkpointing
COLAB_PATH = os.getenv('COLAB_PATH')
CHECKPOINT_PATH = "checkpoints" if COLAB_PATH is None else COLAB_PATH  # "checkpoints/checkpoint_1.tar"
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)

# Set accelerator
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

# Avoid randomness to ensure/improve result reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


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
        optimizer_P.zero_grad()
        optimizer_ET.zero_grad()

        # Freeze P, train E and T together
        P.freeze()
        input_encoded = E.forward(input)
        action_pred = T.forward(input_encoded)
        frozen_privacy_pred = P.forward(input_encoded)
        loss_action = action_loss.forward(action_pred, frozen_privacy_pred, target_action)
        loss_action.backward()
        optimizer_ET.step()

        # Freeze E and T, unfreeze and train P
        P.unfreeze()
        E.freeze()
        T.freeze()
        frozen_input_encoded = E.forward(input)
        privacy_pred = P.forward(frozen_input_encoded)
        loss_privacy = privacy_loss.forward(privacy_pred, target_privacy)
        loss_privacy.backward()
        optimizer_P.step()

        # Unfreeze all models, record losses
        E.unfreeze()
        T.unfreeze()

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
            progress_loader.refresh()
            # print(f"Epoch {epoch+1}/{num_epochs}, Action Loss: {val_loss_action:.4f}, Privacy Loss: {val_loss_privacy:.4f}")
            # print(f"Action accuracy: {val_acc_action:.4f}, Privacy accuracy: {val_acc_privacy:.4f}")

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

if __name__ == "__main__":
    # Specify location of KTH dataset and labels file
    KTH_DATA_DIR = "./KTH"
    KTH_LABELS_DIR = "kth_clips.json"

    # Set parameters according to https://arxiv.org/abs/2208.02459
    num_epochs = 50
    lr = 0.001
    batch_size = 4
    consecutive_frames = 8
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

    # Initialize the BDQEncoder (E), the action attribute predictor (T),
    # and the privacy attribute predictor (P)
    E = BDQEncoder(hardness=5.0).to(device)
    T = ActionRecognitionModel(fine_tune=True, num_classes=6).to(device)
    P = PrivacyAttributePredictor(num_privacy_classes=25).to(device)

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
    criterion_action = ActionLoss(alpha=1)
    criterion_privacy = PrivacyLoss()

    adverserial_training(train_dataloader=train_dataloader, val_dataloader=val_dataloader, E=E, T=T, P=P, 
                         optimizer_ET=optim_ET, optimizer_P=optim_P, scheduler_ET=scheduler_ET, 
                         scheduler_P=scheduler_P, action_loss=criterion_action, privacy_loss=criterion_privacy,
                         last_epoch=last_epoch, num_epochs=num_epochs)