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
    frames = kth[0][0]
    frames = frames.to(dtype=torch.float32)
    frames = frames.unsqueeze(0)
    b, t, c, h, w = frames.shape

    # Load and apply BDQEncoder
    bdq = BDQEncoder()
    frames = bdq(frames)

    frames = frames.squeeze(0)
    # Process and save video with applied BDQ
    video = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h)) #expected W, H
    for frame in frames:
        if frame.dtype != torch.uint8:
            frame = (frame.clamp(0, 1)*255).byte()
        frame = frame.permute(1, 2, 0).contiguous().cpu().numpy() #expected H, W, C
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

    # difference_layer = Difference()
    # bv = torch.randn((1, 3, 1, 4, 4), dtype=torch.float32)
    # d = difference_layer.forward(bv)
    # # m = ActionRecognitionModel(num_classes = 400, fine_tune = True)
    # # print(m.forward(torch.randn((1, 8, 3, 256, 256))).shape)
    # print(d)

if __name__ == '__main__':
    main()