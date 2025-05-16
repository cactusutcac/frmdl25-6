import torch
import json
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import torch.nn as nn

device = "cpu"
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 8
sampling_rate = 8
frames_per_second = 30
clip_duration = (num_frames * sampling_rate) / frames_per_second
start_sec = 0
end_sec = start_sec + clip_duration

class ActionRecognitionModel(nn.Module):
    def __init__(self, fine_tune, num_classes = 400, id_to_classname = None):
        super(ActionRecognitionModel, self).__init__()
        model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
        model = model.eval()
        model = model.to(device)
        if fine_tune:
            model.blocks[-1].proj = nn.Linear(in_features = model.blocks[-1].proj.in_features, out_features = num_classes)
        self.model = model
        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose([UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size = side_size),
                    CenterCropVideo(crop_size = (crop_size, crop_size))])
        )
        self.id_to_classname = id_to_classname

    #accepts [B?, C=3, T=num_frames?, crop_size, crop_size]
    def forward(self, x):
        x = self.model(x)
        return x

    def test(self, video_path):
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec = start_sec, end_sec = end_sec)
        video_data = self.transform(video_data)
        inputs = video_data["video"]
        return self.predict(inputs)

    def predict(self, inputs):
        inputs = inputs.to(device)
        preds = self.forward(inputs[None, ...])
        post_act = torch.nn.Softmax(dim = 1)
        preds = post_act(preds)
        pred_classes = preds.topk(k = 1).indices[0]
        pred_class_names = [self.id_to_classname[int(i)] for i in pred_classes]
        return pred_class_names

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

# Adapted from: https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/