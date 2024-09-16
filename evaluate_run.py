'''
Main script for evaluating temporal attention model
'''

import numpy as np
import pickle
import torch
import torch.nn.functional as F
from temporal_attention_model import TemporalAttentionModel, SubModel


def temporal_pairs(dict, sessions):
    '''
    Create temporal pairs between adjacent windows for all data
    :param dict: dictionary of all session data - each session shape (n_windows, n_channels, n_samples)
    :param sessions: list of session names
    :return x_all: temporal pairs of each session as a list of length n_sessions - each entry shape (n_windows, 256, 2)
    :return y_all: ground truth HR labels as a list of length n_sessions
    :return act_all: activity labels as a list of length n_sessions
    '''

    x_all = []
    y_all = []
    act_all = []

    for s in sessions:

        x = dict[s]['bvp'].squeeze(axis=1)

        # pair adjacent windows (i, i-1)
        x_pairs = (np.expand_dims(x[1:,:],axis=-1) , np.expand_dims(x[:-1,:],axis=-1))
        x_pairs = np.concatenate(x_pairs,axis=-1)
        # results in concatenated pairs of shape (n_windows, n_samples, 2)

        x_all.append(x_pairs)
        y_all.append(dict[s]['label'][1:])
        act_all.append(dict[s]['activity'][1:])

    return x_all, y_all, act_all

def evaluate_model(dict, sessions):
    '''
    ....
    :param dict: dictionary of all session data - each session shape (n_windows, n_channels, n_samples)
    :param sessions: list of session names
    '''

    def NLL(dist, y):
        '''
        Negative log likelihood loss of observation y, given distribution dist
        :param dist: predicted Gaussian distribution
        :param y: ground truth label
        :return: NLL for each window
        '''
        return -dist.log_prob(y)

    std_threshold = 5

    n_epochs = 200
    batch_size = 128

    # evaluate error stats
    error_abs = []
    error_std_thr = []
    activity_errors = np.empty((15, 9))
    activity_errors[:] = np.nan

    pcts_dropped = []                       # percentages dropped
    nll_e = []
    error_vs_std = []

    x, y, act = temporal_pairs(dict, sessions)

    for s in range(0,15):

        n_epochs = 100
        batch_size = 256
        n_ch = 2
        patience = 150

        fs = 32

        # create simple train/test split
        train_idxs = np.array([i for i in range(0,15) if i!=s])

        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)

        X_test = x[s]
        y_test = y[s]
        act_test = act[s]

        ## implement error classifaction on raw (pre-probabilistic) model outputs

        # load trained model for corresponding session
        try:
            checkpoint_path = f'/models/temporal_attention_model_session_S{s+1}.pth'
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['model_state_dict']
        except FileNotFoundError:
            print(f'No pretrained model found for Session S{s+1}')
            break

        # instantiate model with pretrained weights
        model = TemporalAttentionModel()
        model.load_state_dict(state_dict)

        # create submodel that excludes last layer
        submodel = SubModel()
        submodel_state_dict = submodel.state_dict()

        # instantiate submodel with pretrained weights
        state_dict = {k: v for k, v in state_dict.items() if k in submodel_state_dict}
        submodel.load_state_dict(state_dict)

        # evaluate on submodel
        model.eval()
        submodel.eval()

        with torch.no_grad():
            x_cur = X_test[:, :, 0].unsqueeze(1)
            x_prev = X_test[:, :, -1].unsqueeze(1)
            y_pred = submodel(x_cur, x_prev)

            # calculate loss of prediction against probabilistic output
            loss = NLL(y_test, model(x_cur, x_prev)).numpy()
            loss = np.diagonal(loss)

            nll_e.append(loss.mean())

        # extract mean and stdev predictions
        y_pred_m = y_pred[:,0]
        y_pred_std = 1 + F.softplus(y_pred[:,-1])

        # get absolute error between prediction and ground truth
        error = np.mean(np.abs(y_pred_m - y_test))
        error_abs.append(error)

        error_thr = np.mean(np.abs(y_pred_m[y_pred_std < std_threshold] - y_test[y_pred_std < std_threshold]))
        error_std_thr.append(error_thr)

        pct_dropped = np.argwhere(y_pred_std < std_threshold).size / y_test.size
        pcts_dropped.append(pct_dropped)

        # loss





    return



def main():

    def load_dict(filename):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # load x_bvp data
    dict = load_dict(filename='ppg_filt_dict')

    evaluate_model(dict, sessions)

if __name__ == '__main__':
    main()

