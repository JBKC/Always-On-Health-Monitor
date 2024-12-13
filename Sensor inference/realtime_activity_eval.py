'''
Called from ESP32_BT_realtime_activity_inference.py script
'''

import numpy as np
import torch

def run_inference(x, model):

    model.eval()
    x = torch.from_numpy(x).float()

    out = model(x)
    softmax = torch.nn.Softmax(dim=1)
    out = softmax(out)

    return int(torch.argmax(out, dim=1))


def main(x, model):
    '''
    :param x: shape (1,6,256)
    :param model: time series model
    '''

    return run_inference(x, model)


if __name__ == '__main__':
    main()
