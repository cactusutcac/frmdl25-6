import torch

from action_recognition_model import ActionRecognitionModel
from difference import Difference

def main():

    difference_layer = Difference()
    bvj = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    bvi = torch.randn((2, 3, 224, 224))
    difference_layer.forward(bvj)
    d = difference_layer.forward(bvi)
    m = ActionRecognitionModel(num_classes = 400, fine_tune = True)
    print(m.forward(torch.randn((1, 3, 8, 256, 256))).shape)
    # print(d)
    # print(bvi - bvj)

if __name__ == '__main__':
    main()