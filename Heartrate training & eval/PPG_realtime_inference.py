'''
Passes PPG sensor & accelerometer input through trained model to produce real-time heart-rate inference
'''

import torch
from temporal_attention_model import TemporalAttentionModel








def main():

    # initialise model
    checkpoint = torch.load('../models/temporal_attention_model_full_augment_session_S6.pth')
    model = TemporalAttentionModel()  # Update with any required parameters for your model initialization

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

if __name__ == '__main__':
    main()
