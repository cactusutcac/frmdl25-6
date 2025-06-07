import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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

    def forward(self, bdq_encoded_frame):
        """
        Forward pass for the privacy attribute predictor.

        Args:
            bdq_encoded_frame (torch.Tensor): The output from the BDQ encoder.
                Shape: (B, C, H, W), where
                B = batch size
                C = number of channels
                H = height
                W = width

        Returns:
            torch.Tensor: softmax probabilities for privacy attributes.
                          Shape: (B, num_privacy_classes)
        """
        # Get logits from the ResNet feature extractor for all (B) frames
        logits_all_frames = self.resnet_feature_extractor(bdq_encoded_frame) # Shape: (B, num_privacy_classes)

        # Apply softmax to get probabilities for each frame
        softmax_all_frames = F.softmax(logits_all_frames, dim=1) # Shape: (B, num_privacy_classes)

        return softmax_all_frames


if __name__ == "__main__":
    from preprocess import KTHBDQDataset

    # --- Parameters for Person Identification ---
    root_dir = "/datasets/KTH"
    json_path = "/datasets/kth_clips.json"
    clip_len = 16
    resize = (224, 224)
    num_persons = 25  # 25 actors in KTH

    # --- Load a clip and its person label from the dataset ---
    dataset = KTHBDQDataset(
        root_dir=root_dir,
        json_path=json_path,
        clip_len=clip_len,
        resize=resize
    )
    # You need to modify your KTHBDQDataset to also return the person ID (0-24) as the label
    clip = dataset[0]  # shape: [T, C, H, W], person_id: int

    # --- Prepare input for the model ---
    clip = clip.unsqueeze(0)  # [1, T, C, H, W]
    C = clip.shape[2]
    if C == 3:
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, C, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, C, 1, 1)
    elif C == 1:
        imagenet_mean = torch.tensor([0.456]).view(1, 1, C, 1, 1)
        imagenet_std = torch.tensor([0.224]).view(1, 1, C, 1, 1)
    else:
        raise ValueError("Unexpected channel count in clip.")

    normalized_clip = (clip - imagenet_mean) / imagenet_std

    # --- Instantiate and run the person identification model ---
    person_id_model = PrivacyAttributePredictor(
        num_privacy_classes=num_persons,
        pretrained_resnet=True
    )
    person_id_model.eval()

    with torch.no_grad():
        predictions = person_id_model(normalized_clip)  # shape: [1, 25]

    predicted_person = torch.argmax(predictions, dim=1).item()

    print(f"Input video shape: {normalized_clip.shape}")
    print(f"Person ID predictions shape: {predictions.shape}")  # (1, 25)
    print(f"Predicted person ID: {predicted_person}")
    # print(f"True person ID: {person_id}")