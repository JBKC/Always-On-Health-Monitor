'''
Passes PPG sensor & accelerometer input through trained model to produce real-time heart-rate inference
'''

import torch

def main():
    # extract model
    checkpoint_path = '../models/temporal_attention_model_full_augment_session_S6.pth'
    checkpoint = torch.load(checkpoint_path)

    print("Keys in the checkpoint file:")
    print(checkpoint.keys())


if __name__ == '__main__':
    main()
