'''
Runs inference on single test session for equivalent trained model
'''

import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
from temporal_attention_model import TemporalAttentionModel
from torch.distributions import Normal
import matplotlib.pyplot as plt


def temporal_pairs(dict, eval_session):
    '''
    Create temporal pairs between adjacent windows for all data
    :param dict: dictionary of all session data - each session shape (n_windows, n_channels, n_samples)
    :param eval_session: session name (string)
    :return x_all: array of temporal pairs for current session - each entry shape (n_windows, 256, 2)
    :return y_all: ground truth HR labels as an array of shape (n_windows,)
    '''

    x = dict[eval_session]['bvp'].squeeze(axis=1)

    # pair adjacent windows (i, i-1)
    x_pairs = (np.expand_dims(x[1:,:],axis=-1) , np.expand_dims(x[:-1,:],axis=-1))
    x_all = np.concatenate(x_pairs,axis=-1)     # concatenated pairs of shape (n_windows, n_samples, 2)

    y_all = dict[eval_session]['label'][1:]

    return x_all, y_all

def inference(dict, eval_session):
    '''
    :param dict: dictionary of all session data - each session shape (n_windows, n_channels, n_samples)
    :param eval_session: session name (string)
    '''

    # create temporal pairs for model
    x, y = temporal_pairs(dict, eval_session)

    print(f'Evaluating Session {eval_session}')

    # create simple test split
    X_test = torch.from_numpy(x).float()
    y_test = torch.from_numpy(y).float()

    # load trained model for selected session
    try:
        # checkpoint_path = f'../models/temporal_attention_model_full_augment_session_{eval_session}.pth'
        checkpoint_path = f'../models/temporal_attention_model_session_{eval_session}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['model_state_dict']
        for key, value in checkpoint.items():
            print(key, value)

    except FileNotFoundError:
        print(f'No pretrained model found for Session {eval_session}')

    # instantiate model with pretrained weights (exclude probability layer)
    model = TemporalAttentionModel()
    state_dict = {k: v for k, v in state_dict.items() if k in state_dict}
    model.load_state_dict(state_dict)
    model.eval()

    ### perform inference
    with torch.no_grad():
        x_cur = X_test[:, :, 0].unsqueeze(1)
        x_prev = X_test[:, :, -1].unsqueeze(1)

        gaussian = model(x_cur, x_prev)             # returns gaussian object
        hrs = gaussian.mean

        print(f"Heart rate prediction: {hrs}")
        print(f"Heart rate labels: {y_test}")

        # plotting
        time = np.linspace(0, 2 * len(y_test), len(y_test))
        plt.plot(time, y_test, color='black',linewidth=1, label='Labels')
        plt.plot(time, hrs, color='red', linewidth=1, label='Predictions')
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Heart Rate (BPM)")
        plt.show()

    return


def main():

    def load_dict(filename):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    # enter model's test session
    eval_session = 'S7'

    dict = load_dict(filename='../ppg_filt_dict')
    inference(dict, eval_session)

if __name__ == '__main__':
    main()

