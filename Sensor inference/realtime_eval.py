import numpy as np
import torch
from Heartrate_training_eval.temporal_attention_model import TemporalAttentionModel


def run_inference(x_input, model):

    model.eval()

    x_input = torch.from_numpy(x_input).float()

    with torch.no_grad():
        x_cur = x_input[:, :, 0].unsqueeze(1)           # Current window
        x_prev = x_input[:, :, -1].unsqueeze(1)         # Previous window

        # Perform inference
        with torch.no_grad():
            pred = model(x_cur, x_prev).mean
            print(f"Heart rate prediction: {pred}")

            return pred

def main(buffer, model):

    ### temporary - convert the deque into 2 NumPy arrays each with shape (256, 1) - previous and current 8-second portions
    window_1 = buffer[64:,0]
    window_2 = buffer[:256,0]

    # reshape for model input
    x_input = np.stack((window_1, window_2), axis=-1)
    x_input = x_input[np.newaxis, :]
    print(x_input.shape)              # Shape: (1, 256, 2)

    # Run inference
    run_inference(x_input, model)


if __name__ == '__main__':
    main()
