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

        # Replace the final fully connected layer for the new number of privacy classes
        num_ftrs = self.resnet_feature_extractor.fc.in_features
        self.resnet_feature_extractor.fc = nn.Linear(num_ftrs, num_privacy_classes)

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

if __name__ == "__main__":
    from preprocess import KTHBDQDataset

    # --- Parameters for Person Identification ---
    root_dir = "KTH"
    json_path = "kth_clips.json"
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