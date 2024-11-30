'''
Main script for calculating error metrics of temporal attention model
'''

import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
from temporal_attention_model import TemporalAttentionModel, SubModel
from torch.distributions import Normal


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


    eval_dict = {f'{session}': {} for session in sessions}

    mean_thr = 10                           # threshold for error classification outlined in KID-PPG
    std_thr = 2                             # threshold for submodel_error_thr and pcts_kept

    # create temporal pairs for model
    x, y, act = temporal_pairs(dict, sessions)

    for s, session in enumerate(sessions):

        print(f'Evaluating Session {session}')

        # create simple test split
        X_test = torch.from_numpy(x[s]).float()
        y_test = torch.from_numpy(y[s]).float()
        act_test = act[s]

        # load trained model for corresponding session
        try:
            # checkpoint_path = f'../models/temporal_attention_model_full_augment_session_S6.pth'
            checkpoint_path = f'../dummy_models/temporal_attention_model_session_S{s + 1}.pth'
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['model_state_dict']
        except FileNotFoundError:
            print(f'No pretrained model found for Session S{s+1}')
            break

        # instantiate model with pretrained weights
        model = TemporalAttentionModel()
        state_dict = {k: v for k, v in state_dict.items() if k in state_dict}
        model.load_state_dict(state_dict)
        model.eval()

        # instantiate submodel (that excludes final layer) with pretrained weights
        submodel = SubModel()
        submodel_state_dict = submodel.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in submodel_state_dict}
        submodel.load_state_dict(state_dict)
        submodel.eval()

        # ** perform evaluation - calculate various error metrics **

        with torch.no_grad():
            x_cur = X_test[:, :, 0].unsqueeze(1)
            x_prev = X_test[:, :, -1].unsqueeze(1)

            # 1. calculate loss of probabilistic model against ground truth (average across all windows)
            model_loss = NLL(model(x_cur, x_prev), y_test).mean().numpy()
            eval_dict[session]['model_loss'] = model_loss
            print(f'model_loss: {model_loss}')

            # 2. find probability that submodel mean lies within a threshold on the predicted distribution
            y_pred = submodel(x_cur, x_prev)
            y_pred_m = y_pred[:,0]                                # mean
            y_pred_std = 0.1 + F.softplus(y_pred[:,-1])           # standard deviation
            dist = Normal(loc=y_pred_m, scale=y_pred_std)
            upper = dist.cdf(y_pred_m + mean_thr)
            lower = dist.cdf(y_pred_m - mean_thr)
            p_error = (upper - lower).mean().item()
            eval_dict[session]['p_error'] = p_error
            print(f'p_error: {p_error}')

            # 3. calculate absolute error of submodel (mean vs. ground truth)
            submodel_error = np.mean(np.abs(y_pred_m - y_test).numpy())
            eval_dict[session]['submodel_error'] = submodel_error
            print(f'submodel_error: {submodel_error}')


            # 4. calculate absolute error of submodel for low uncertainty predictions
            submodel_error_thr = np.mean(np.abs(y_pred_m[y_pred_std < std_thr] - y_test[y_pred_std < std_thr]).numpy())
            eval_dict[session]['submodel_error_thr'] = submodel_error_thr
            print(f'submodel_error_thr: {submodel_error_thr}')

            # 5. record how many low uncertainty predictions there are as a % of all predictions
            pct_kept = np.mean(y_pred_std.numpy() < std_thr)
            eval_dict[session]['pct_kept'] = pct_kept
            print(f'pct_kept: {pct_kept}')

            # 6. repeat low uncertainty analysis for a range of thresholds
            error_vs_thr = {}
            for thr in np.arange(1, 3, 0.1):
                thr = round(thr, 1)
                error_vs_thr[thr] = np.mean(np.abs(y_pred_m[y_pred_std < thr] - y_test[y_pred_std < thr]).numpy())
            eval_dict[session]['error_vs_thr'] = error_vs_thr
            print(f'error_vs_thr: {error_vs_thr}')

            # 7. get individual errors for each activity
            activity_error = []
            for activity in np.unique(act_test):
                activity_error = np.mean(np.abs(y_pred_m[act_test == activity] - y_test[act_test == activity]).numpy())
            eval_dict[session]['activity_error'] = activity_error
            print(f'activity_error: {activity_error}')

    # save dictionary
    output_dir = (f'./evaluation_results/')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'temporal_attention_model_full_augment_eval_dict.pkl')

    with open(output_path, 'wb') as file:
        pickle.dump(eval_dict, file)
    print(f'Data dictionary saved to {output_path}')

    return


def main():

    def load_dict(filename):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # load x_bvp data
    dict = load_dict(filename='../ppg_filt_dict')

    evaluate_model(dict, sessions)

if __name__ == '__main__':
    main()

