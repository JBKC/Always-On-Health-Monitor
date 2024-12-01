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
        print(f"BPM: {hr_pred:.4f}")

        return hr_pred

def main(buffer, model):

    # reshape for model input (x_cur, x_prev) - ie. lastmost and firstmost 8 second portions of the 10 second window
    x_input = np.stack((buffer[64:,0], buffer[:256,0]), axis=-1)
    x_input = x_input[np.newaxis, :]
    # print(x_input.shape)              # Shape: (1, 256, 2)

    # Run inference
    run_inference(x_input, model)


if __name__ == '__main__':
    main()
