import torch
from torch import nn
import torch.nn.functional as F

def validate_frozen_bdq(E, train_dataloader, test_dataloader, device):
    # Freeze BDQ encoder
    E.eval()
    for param in E.parameters():
        param.requires_grad = False

    T = ActionRecognitionModel(fine_tune=True, num_classes=6).to(device)
    P = PrivacyAttributePredictor(num_privacy_classes=25).to(device)

    opt_action = torch.optim.SGD(T.parameters(), lr=0.001, momentum=0.9)
    opt_privacy = torch.optim.SGD(P.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(50):
        T.train()
        P.train()
        total_action_loss = 0.0
        total_privacy_loss = 0.0

        for video, action_label, identity_label in train_dataloader:
            x = video.to(device)  # [B, T, C, H, W]
            y_action = action_label.to(device)
            y_privacy = identity_label.to(device)

            with torch.no_grad():
                bdq = E(x)  # [B, 24, T, H, W]

            # Reshape: 24 channels -> 3 channels (average 8 bins per channel)
            B, _, T_, H, W = bdq.shape
            bdq = bdq.view(B, 3, 8, T_, H, W).mean(dim=2)  # [B, 3, T, H, W]

            # Upsample back to (T=24, H=224, W=224) for I3D compatibility
            bdq = F.interpolate(bdq, size=(24, 224, 224), mode="trilinear", align_corners=False)

            # Pass into T: expects [B, T, C, H, W]
            input_action = bdq.permute(0, 2, 1, 3, 4)
            probs_action = T(input_action)
            loss_action = F.cross_entropy(probs_action, y_action)
            opt_action.zero_grad()
            loss_action.backward()
            opt_action.step()

            # Privacy model: avg over time -> [B, 3, H, W]
            input_privacy = bdq.mean(dim=2)
            probs_privacy = P(input_privacy)
            loss_privacy = F.cross_entropy(probs_privacy, y_privacy)
            opt_privacy.zero_grad()
            loss_privacy.backward()
            opt_privacy.step()

            total_action_loss += loss_action.item()
            total_privacy_loss += loss_privacy.item()

        print(f"[Epoch {epoch+1}] Action Loss: {total_action_loss:.3f} | Privacy Loss: {total_privacy_loss:.3f}")

    # Evaluation
    T.eval()
    P.eval()
    correct_action = 0
    correct_privacy = 0
    total = 0

    with torch.no_grad():
        for video, action_label, identity_label in test_dataloader:
            x = video.to(device)
            y_action = action_label.to(device)
            y_privacy = identity_label.to(device)

            bdq = E(x)
            bdq = bdq.view(bdq.size(0), 3, 8, bdq.size(2), bdq.size(3), bdq.size(4)).mean(dim=2)
            bdq = F.interpolate(bdq, size=(24, 224, 224), mode="trilinear", align_corners=False)

            input_action = bdq.permute(0, 2, 1, 3, 4)
            input_privacy = bdq.mean(dim=2)

            probs_action = T(input_action)
            probs_privacy = P(input_privacy)

            correct_action += (probs_action.argmax(dim=1) == y_action).sum().item()
            correct_privacy += (probs_privacy.argmax(dim=1) == y_privacy).sum().item()
            total += x.size(0)

    print("\n[Validation Results with frozen BDQ encoder]")
    print(f"Action Accuracy:  {correct_action / total * 100:.2f}%")
    print(f"Privacy Accuracy: {correct_privacy / total * 100:.2f}%")
