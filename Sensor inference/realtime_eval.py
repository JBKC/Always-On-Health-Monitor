'''
Called from PPG_realtime_inference.py script
'''

import numpy as np
import torch

def run_inference(x_input, model):

    model.eval()

    x_input = torch.from_numpy(x_input).float()

    with torch.no_grad():
        x_cur = x_input[:, :, 0].unsqueeze(1)           # Current window
        x_prev = x_input[:, :, -1].unsqueeze(1)         # Previous window

        hr_pred = model(x_cur, x_prev).mean.numpy()[0]         # model returns gaussian

        return hr_pred

def main(x, model):
    '''
    :param x: shape (2,1,256) = x_cur and x_prev pairing of windows
    :param model: attention-based model
    '''

    # reshape for model input (1,256,2)
    x = np.transpose(x, (1, 2, 0))

    return run_inference(x, model)


if __name__ == '__main__':
    main()
