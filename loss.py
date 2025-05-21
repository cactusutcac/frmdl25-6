import torch
from torch import nn


class ActionLoss(nn.Module):
    """
    Args:
        encoder: the BDQ encoder
        target_predictor: 3D CNN N for predicting target action attribute
        alpha: the adversarial weight for trade-off between action and privacy recognition
    """
    def __init__(self, encoder, target_predictor, alpha=1):
        super().__init__()
        self.encoder = encoder
        self.target_predictor = target_predictor
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def entropy(self, x, dim=1, eps=1e-6):
        x = torch.clamp(x, eps)
        return -torch.sum(x * torch.log(x), dim=dim)

    """
    Args:
        V: input (video);
        L_action: the ground-truct action labels of the inputs
        fixed_privacy_predictor: a 2D CNN for predicting the privacy attribute (with frozen weights)
    """
    def forward(self, V, L_action, fixed_privacy_predictor):
        V_encoded = self.encoder(V)
        y_pred = self.target_predictor(V_encoded)
        loss = self.cross_entropy(y_pred, L_action) - self.alpha * self.entropy(fixed_privacy_predictor(V_encoded))
        return loss

class PrivacyLoss(nn.Module):
    """
    Args:
        privacy_predictor: 2D CNN for predicting the privacy attribute
    """
    def __init__(self, privacy_predictor):
        super().__init__()
        self.privacy_predictor = privacy_predictor
        self.cross_entropy = nn.CrossEntropyLoss()

    """
    Args:
        V: the input (video)
        L_privacy: the ground-truth privacy labels
        fixed_encoder: the (fixed) BDQ encoder
    """
    def forward(self, V, L_privacy, fixed_encoder):
        V_encoded = fixed_encoder(V)
        y_pred = self.privacy_predictor(V_encoded)
        loss = self.cross_entropy(y_pred, L_privacy)
        return loss
