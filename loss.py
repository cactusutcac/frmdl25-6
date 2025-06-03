import torch
from torch import nn

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
