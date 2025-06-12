import os
import random
import numpy as np
import torch
from tqdm import tqdm

from bdq_encoder.BDQ import BDQEncoder
from action_recognition_model import ActionRecognitionModel
from loss import ActionLoss, PrivacyLoss
from preprocess import KTHBDQDataset, IXMASBDQDataset, ConsecutiveTemporalSubsample, MultiScaleCrop, NormalizePixelValues, NormalizeVideo
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

import argparse

from visualization.quantization_steps import save_quantizer_mapping 

# Setup checkpointing
COLAB_PATH = os.getenv('COLAB_PATH')
CHECKPOINT_PATH = "checkpoints" if COLAB_PATH is None else COLAB_PATH  # "checkpoints/checkpoint_1.tar"
MODE_ACTION = "action"
MODE_PRIVACY = "privacy"
CHECKPOINT_PATH_ACTION = os.path.join(CHECKPOINT_PATH, MODE_ACTION)
CHECKPOINT_PATH_PRIVACY = os.path.join(CHECKPOINT_PATH, MODE_PRIVACY)
for checkpoint_path in [CHECKPOINT_PATH, CHECKPOINT_PATH_ACTION, CHECKPOINT_PATH_PRIVACY]:
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

# Set accelerator
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

# Avoid randomness to ensure/improve result reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

def delete_old_checkpoints(checkpoint_path: str):
    checkpoints = get_sorted_checkpoints(checkpoint_path)
    if len(checkpoints) > 2:
        for file, _ in checkpoints[:-2]:
            os.remove(file)

def compute_accuracy(input, target_action, target_privacy, E, T, P, random_frame: int | None = None):
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
        privacy_acc = 0.0
        frames = range(input_encoded.size(1)) if random_frame is None else [random_frame] #T
        for frame in frames:
            P_pred = P.forward(input_encoded[:, frame, :, :, :]).argmax(dim=1)
            privacy_acc += torch.sum(P_pred == target_privacy)

        action_acc = torch.sum(T_pred == target_action)
        privacy_acc /= len(frames)

        return action_acc, privacy_acc

def train_once(train_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor, 
               action_loss: ActionLoss, privacy_loss: PrivacyLoss, optimizer_ET: Optimizer, optimizer_P: Optimizer):
    """
    Function to perform one training epoch of adversarial training from https://arxiv.org/abs/2208.02459
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
        optimizer_ET.zero_grad()

        # Freeze P, train E and T together
        input_encoded = E.forward(input)
        action_pred = T.forward(input_encoded)
        # Pick random frame for 2D privacy predictor
        random_frame = random.randint(0, input_encoded.size(1) - 1)
        frozen_privacy_pred = P.forward(input_encoded[:, random_frame, :, :, :])
        loss_action = action_loss.forward(action_pred, frozen_privacy_pred, target_action)
        loss_action.backward()
        optimizer_ET.step()

        optimizer_P.zero_grad()
        # Freeze E and T, unfreeze and train P
        frozen_input_encoded = E.forward(input)
        privacy_pred = P.forward(frozen_input_encoded[:, random_frame, :, :, :])
        loss_privacy = privacy_loss.forward(privacy_pred, target_privacy)
        loss_privacy.backward()
        optimizer_P.step()

        # Unfreeze all models, record losses
        # Compute statistics
        acc_action, acc_privacy = compute_accuracy(input, target_action, target_privacy, E, T, P, random_frame)

        total_loss_action += loss_action.item()
        total_loss_privacy += loss_privacy.item()

        total_acc_action += acc_action.item()
        total_acc_privacy += acc_privacy.item()

    # Average out accuracies
    total_acc_action /= len(train_dataloader.dataset)
    total_acc_privacy /= len(train_dataloader.dataset)

    return total_loss_action, total_loss_privacy, total_acc_action, total_acc_privacy

def validate_once(val_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor, 
                  loss_f: nn.CrossEntropyLoss):
    """
    Function to perform one validation epoch of adversarial training from https://arxiv.org/abs/2208.02459
    Args:
        val_dataloader: DataLoader for the validation split of the dataset
        E: the BDQ encoder
        T: 3d resnet50 for predicting target action attributes
        P: 2d resnet50 for predicting target privacy attributes
        loss_f: criterion for optimizing attribute prediction
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
            privacy_preds = []
            frames = input_encoded.size(1) # T
            for frame in range(frames):
                privacy_preds.append(P.forward(input_encoded[:, frame, :, :, :]))

            # Compute statistics
            loss_action = loss_f.forward(action_pred, target_action)
            loss_privacy = 0
            for frame in range(frames):
                privacy_pred = privacy_preds[frame]
                loss_privacy += loss_f.forward(privacy_pred, target_privacy)
            loss_privacy /= frames

            acc_action, acc_privacy = compute_accuracy(input, target_action, target_privacy, E, T, P)

            total_loss_action += loss_action.item()
            total_loss_privacy += loss_privacy.item()
            total_acc_action += acc_action.item()
            total_acc_privacy += acc_privacy.item()

        # Average out accuracies
        total_acc_action /= len(val_dataloader.dataset)
        total_acc_privacy /= len(val_dataloader.dataset)

        return total_loss_action, total_loss_privacy, total_acc_action, total_acc_privacy

def adversarial_training(train_dataloader: DataLoader, val_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel,
                         P: PrivacyAttributePredictor, optimizer_ET: Optimizer, optimizer_P: Optimizer, scheduler_ET: LRScheduler, 
                         scheduler_P: LRScheduler, action_loss: ActionLoss, privacy_loss: PrivacyLoss, writer: SummaryWriter, cross_entropy: nn.CrossEntropyLoss, last_epoch=0, num_epochs=20):#fixme
    """
    Function encapsulating the whole adversarial training process from https://arxiv.org/abs/2208.02459.
    If last_epoch >= num_epochs then only runs validation once.
    Args:
        train_dataloader: DataLoader for the training split of the dataset
        val_dataloader: DataLoader for the validation split of the dataset
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
        delete_old_checkpoints(CHECKPOINT_PATH)

    if last_epoch >= num_epochs:
        val_loss_action, val_loss_privacy, val_acc_action, val_acc_privacy = validate_once(val_dataloader=val_dataloader, E=E, T=T, P=P, 
                                                                                            loss_f=cross_entropy)
        print(f"Action accuracy: {val_acc_action:.4f}, Privacy accuracy: {val_acc_privacy:.4f}")
        print(f"Action Loss: {val_loss_action:.4f}, Privacy Loss: {val_loss_privacy:.4f}")

    with tqdm(range(last_epoch, num_epochs), total=num_epochs, initial=last_epoch, desc="Adversarial training", unit="epoch", position=0, leave=True) as progress_loader:
        for epoch in progress_loader:
            train_loss_action, train_loss_privacy, train_acc_action, train_acc_privacy = train_once(train_dataloader=train_dataloader, E=E, T=T, P=P, 
                                                                                                    action_loss=action_loss, privacy_loss=privacy_loss, 
                                                                                                    optimizer_ET=optimizer_ET, optimizer_P=optimizer_P)
            
            val_loss_action, val_loss_privacy, val_acc_action, val_acc_privacy = validate_once(val_dataloader=val_dataloader, E=E, T=T, P=P, 
                                                                                            loss_f=cross_entropy)

            # Update learning rates
            scheduler_ET.step()
            scheduler_P.step()
            save_checkpoint(epoch + 1)
            writer.add_scalars("Loss", {'train_loss_action': train_loss_action,
                                        'train_loss_privacy': train_loss_privacy,
                                        'val_loss_action': val_loss_action,
                                        'val_loss_privacy': val_loss_privacy}, epoch)
            writer.add_scalars('Accuracy', {'train_acc_action': train_acc_action,
                                        'train_acc_privacy': train_acc_privacy,
                                        'val_acc_action': val_acc_action,
                                        'val_acc_privacy': val_acc_privacy}, epoch)

            # Display statistics
            progress_loader.set_postfix(action_loss=val_loss_action.numpy(), privacy_loss=val_loss_privacy.numpy(),
                                         action_accuracy=val_acc_action.numpy(), privacy_accuracy= val_acc_privacy.numpy())
            progress_loader.refresh()

def train_once_resnet(train_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor,
                      loss_f: nn.CrossEntropyLoss, optimizer: Optimizer, mode: str):
    """
    Function to perform one training epoch of validation training from https://arxiv.org/abs/2208.02459
    Args:
        train_dataloader: DataLoader for the training split of the dataset
        E: the BDQ encoder
        T: 3d resnet50 for predicting target action attributes
        P: 2d resnet50 for predicting target privacy attributes
        loss_f: criterion for optimizing attribute prediction
        optimizer: SGD optimizer for the attribute predictor
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
        optimizer.zero_grad()

        # train T or P
        with torch.no_grad():
            frozen_input_encoded = E.forward(input)
        action_pred = T.forward(frozen_input_encoded)
        # Pick random frame for 2D privacy predictor
        random_frame = random.randint(0, frozen_input_encoded.size(1) - 1)
        privacy_pred = P.forward(frozen_input_encoded[:, random_frame, :, :, :])
        loss_action = loss_f.forward(action_pred, target_action)
        loss_privacy = loss_f.forward(privacy_pred, target_privacy)
        if mode == 'action':
            loss = loss_action
        elif mode == 'privacy':
            loss = loss_privacy
        loss.backward()
        optimizer.step()

        # record losses
        # Compute statistics
        acc_action, acc_privacy = compute_accuracy(input, target_action, target_privacy, E, T, P, random_frame)

        total_loss_action += loss_action.item()
        total_loss_privacy += loss_privacy.item()

        total_acc_action += acc_action.item()
        total_acc_privacy += acc_privacy.item()

    # Average out accuracies
    total_acc_action /= len(train_dataloader.dataset)
    total_acc_privacy /= len(train_dataloader.dataset)

    return total_loss_action, total_loss_privacy, total_acc_action, total_acc_privacy

def resnet_training(train_dataloader: DataLoader, val_dataloader: DataLoader, E: BDQEncoder, T: ActionRecognitionModel,
                    P: PrivacyAttributePredictor, optimizer: Optimizer, scheduler: LRScheduler,
                    loss_f: nn.CrossEntropyLoss, writer: SummaryWriter, mode: str, last_epoch=0, num_epochs=20):#fixme
    """
    Function encapsulating the whole validation training process from https://arxiv.org/abs/2208.02459.
    Args:
        train_dataloader: DataLoader for the training split of the dataset
        val_dataloader: DataLoader for the validation split of the dataset
        E: the BDQ encoder
        T: 3d resnet50 for predicting target action attributes
        P: 2d resnet50 for predicting target privacy attributes
        optimizer: SGD optimizer for the attribute predictor
        scheduler: learning rate scheduler for updating learning rate each epoch for optimizer
        loss_f: criterion for optimizing attribute prediction
        last_epoch (optional, int): checkpoint of last saved epoch
        num_epochs (optional, int): number of epochs to train for (default=50)
    """
    def save_checkpoint(epoch: int):
        torch.save({
            'T_state_dict': T.state_dict(),
            'P_state_dict': P.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(CHECKPOINT_PATH, mode, f"checkpoint_{epoch}.tar"))
        delete_old_checkpoints(os.path.join(CHECKPOINT_PATH, mode))

    with tqdm(range(last_epoch, num_epochs), total=num_epochs, initial=last_epoch, desc=f"{mode} ResNet training", unit="epoch", position=0, leave=True) as progress_loader:
        for epoch in progress_loader:
            train_loss_action, train_loss_privacy, train_acc_action, train_acc_privacy = train_once_resnet(train_dataloader=train_dataloader, E=E, T=T, P=P,
                                                                                                           loss_f=loss_f, optimizer=optimizer, mode=mode)

            val_loss_action, val_loss_privacy, val_acc_action, val_acc_privacy = validate_once(val_dataloader=val_dataloader, E=E, T=T, P=P,
                                                                                            loss_f=loss_f)

            # Update learning rates
            scheduler.step()
            save_checkpoint(epoch + 1)
            writer.add_scalars(f"Loss_{mode}", {'train_loss_action': train_loss_action,
                                        'train_loss_privacy': train_loss_privacy,
                                        'val_loss_action': val_loss_action,
                                        'val_loss_privacy': val_loss_privacy}, epoch)
            writer.add_scalars(f'Accuracy_{mode}', {'train_acc_action': train_acc_action,
                                        'train_acc_privacy': train_acc_privacy,
                                        'val_acc_action': val_acc_action,
                                        'val_acc_privacy': val_acc_privacy}, epoch)

            # Display statistics
            progress_loader.set_postfix(action_loss=val_loss_action.numpy(), privacy_loss=val_loss_privacy.numpy(),
                                         action_accuracy=val_acc_action.numpy(), privacy_accuracy=val_acc_privacy.numpy())
            progress_loader.refresh()

def load_train_checkpoint(E: BDQEncoder, T: ActionRecognitionModel, P: PrivacyAttributePredictor,
               optim_ET: Optimizer, optim_P: Optimizer, scheduler_ET: LRScheduler, scheduler_P: LRScheduler, PATH: str | None):
    if PATH is None:
        return
    checkpoint = torch.load(PATH, weights_only=True, map_location=torch.device(device))
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

def load_resnet_train_checkpoint(T: ActionRecognitionModel, P: PrivacyAttributePredictor,
               optim: Optimizer, scheduler: LRScheduler, PATH: str | None):
    if PATH is None:
        return
    checkpoint = torch.load(PATH, weights_only=True, map_location=torch.device(device))
    T.load_state_dict(checkpoint['T_state_dict'])
    T.to(device)
    P.load_state_dict(checkpoint['P_state_dict'])
    P.to(device)
    optim.load_state_dict(checkpoint['optim_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

def main(dataset):
    # Specify location of datasets and labels file
    KTH_DATA_DIR = "./datasets/KTH"
    KTH_LABELS_DIR = "./datasets/kth_clips.json"
    IXMAS_DATA_DIR = './datasets/IXMAS'
    IXMAS_LABELS_DIR = './datasets/ixmas_clips_6.json'

    # Set parameters according to https://arxiv.org/abs/2208.02459
    num_epochs = 20 #fixme
    lr = 0.001
    batch_size = 4
    consecutive_frames = 24 # Not 32 due to hardware limitation 
    crop_size = (224, 224)
    writer = SummaryWriter(log_dir=os.path.join(CHECKPOINT_PATH, "runs"))

    # Difine transformation sequence according to Section 4.2 in https://arxiv.org/abs/2208.02459
    train_transform = Compose([
        ConsecutiveTemporalSubsample(consecutive_frames), # first, sample 32 consecutive frames
        MultiScaleCrop(), # then, apply randomized multi-scale crop
        Resize(crop_size), # then, resize to (224, 224)
        NormalizePixelValues(), # (also normalize pixel values for pytorch)
        NormalizeVideo()
    ])
    val_transform = Compose([
        ConsecutiveTemporalSubsample(consecutive_frames), # first sample 32 consecutive frames
        CenterCrop(crop_size),  # then, we apply a center crop of (224, 224) without scaling (resizing)
        NormalizePixelValues(), # (also normalize pixel values for pytorch)
        NormalizeVideo()
    ])

    # Dynamically load dataset
    if dataset == 'kth':
        train_data = KTHBDQDataset(
            root_dir=KTH_DATA_DIR,
            json_path=KTH_LABELS_DIR,
            transform=train_transform,
            split="train",
        )
        val_data = KTHBDQDataset(
            root_dir=KTH_DATA_DIR,
            json_path=KTH_LABELS_DIR,
            transform=val_transform,
            split="val",
        )
    elif dataset == 'ixmas':
        print(f"IXMAS_LABELS_DIR used: {IXMAS_LABELS_DIR}")
        train_data = IXMASBDQDataset(
            root_dir=IXMAS_DATA_DIR,
            json_path=IXMAS_LABELS_DIR,
            transform=train_transform,
            split="train",
        )
        val_data = IXMASBDQDataset(
            root_dir=IXMAS_DATA_DIR,
            json_path=IXMAS_LABELS_DIR,
            transform=val_transform,
            split="val",
        )
    
    # Wrap in DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=4,
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
    optim_P = SGD(params=P.parameters(), lr=lr)
    scheduler_ET = CosineAnnealingLR(optimizer=optim_ET, T_max=num_epochs)
    scheduler_P = CosineAnnealingLR(optimizer=optim_P, T_max=num_epochs)
    checkpoints = get_sorted_checkpoints(CHECKPOINT_PATH)
    last_checkpoint_path = None
    last_epoch = 0
    if len(checkpoints) > 0:
        last_checkpoint_path, last_epoch = checkpoints[-1]
    load_train_checkpoint(E, T, P, optim_ET, optim_P, scheduler_ET, scheduler_P, last_checkpoint_path)
    criterion_action = ActionLoss(alpha=1)
    criterion_privacy = PrivacyLoss()
    cross_entropy = nn.CrossEntropyLoss().to(device)

    adversarial_training(train_dataloader=train_dataloader, val_dataloader=val_dataloader, E=E, T=T, P=P,
                         optimizer_ET=optim_ET, optimizer_P=optim_P, scheduler_ET=scheduler_ET, 
                         scheduler_P=scheduler_P, action_loss=criterion_action, privacy_loss=criterion_privacy,
                         writer=writer, cross_entropy=cross_entropy, last_epoch=last_epoch, num_epochs=num_epochs)

    # Re-initialize the action attribute predictor (T),
    # and the privacy attribute predictor (P)
    T = ActionRecognitionModel(fine_tune=True, num_classes=6).to(device)
    P = PrivacyAttributePredictor(num_privacy_classes=25).to(device)

    # Initialize optimizer and scheduler
    optim = SGD(params=T.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer=optim, T_max=num_epochs)
    checkpoints = get_sorted_checkpoints(CHECKPOINT_PATH_ACTION)
    last_checkpoint_path = None
    last_epoch = 0
    if len(checkpoints) > 0:
        last_checkpoint_path, last_epoch = checkpoints[-1]
    load_resnet_train_checkpoint(T=T, P=P, optim=optim, scheduler=scheduler, PATH=last_checkpoint_path)

    resnet_training(train_dataloader=train_dataloader, val_dataloader=val_dataloader, E=E, T=T, P=P,
                    optimizer=optim, scheduler=scheduler,
                    loss_f=cross_entropy,
                    writer=writer, mode=MODE_ACTION, last_epoch=last_epoch, num_epochs=num_epochs)

    # Initialize optimizer and scheduler
    optim = SGD(params=P.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer=optim, T_max=num_epochs)
    checkpoints = get_sorted_checkpoints(CHECKPOINT_PATH_PRIVACY)
    last_checkpoint_path = None
    last_epoch = 0
    if len(checkpoints) > 0:
        last_checkpoint_path, last_epoch = checkpoints[-1]
    load_resnet_train_checkpoint(T=T, P=P, optim=optim, scheduler=scheduler, PATH=last_checkpoint_path)

    resnet_training(train_dataloader=train_dataloader, val_dataloader=val_dataloader, E=E, T=T, P=P,
                    optimizer=optim, scheduler=scheduler,
                    loss_f=cross_entropy,
                    writer=writer, mode=MODE_PRIVACY, last_epoch=last_epoch, num_epochs=num_epochs)
    writer.flush()
    writer.close()

    # Save quantizer curve after training 
    save_quantizer_mapping(E.encoder[2], f"quant_steps_{dataset}.csv", device="cuda")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BDQ model on selected dataset. ")
    parser.add_argument('--dataset', type=str, choices=['kth', 'ixmas'], required=True,
                        help='Dataset to use: "KTH" or "IXMAS"')
    args = parser.parse_args()
    main(args.dataset)
