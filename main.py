import cv2
import torch

from action_recognition_model import ActionRecognitionModel
from difference import Difference
from BDQ import BDQEncoder
from preprocess import KTHBDQDataset

def main():
    # Load dataset
    kth = KTHBDQDataset('./KTH', 'kth_clips.json')
    
    # Load example video
    frames, info = kth[0]
    t, c, h, w = frames.shape
    
    # Load and apply BDQEncoder
    bdq = BDQEncoder()
    frames = bdq(frames)

    # Process and save video with applied BDQ
    video = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    for frame in frames:
        if frame.dtype != torch.uint8:
            frame = (frame.clamp(0, 1)*255).byte()
        frame = frame.permute(1, 2, 0).contiguous().cpu().numpy()
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    # difference_layer = Difference()
    # bvj = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    # bvi = torch.randn((2, 3, 224, 224))
    # difference_layer.forward(bvj)
    # d = difference_layer.forward(bvi)
    # m = ActionRecognitionModel(num_classes = 400, fine_tune = True)
    # print(m.forward(torch.randn((1, 3, 8, 256, 256))).shape)
    # print(d)
    # print(bvi - bvj)

if __name__ == '__main__':
    main()