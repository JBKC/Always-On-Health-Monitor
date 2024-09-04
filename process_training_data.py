'''
Initial file for pulling and processing training data from PPG-DaLiA dataset
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from accelerometer_cnn import AdaptiveLinearModel

def save_data(s, data_dict):
    '''
    Pull raw data from PPG Dalia files and save down to a dictionary
    :param s: session name
    :param data_dict: empty dictionary to hold data
    :return:
    '''

    # pull raw data
    with open(f'ppg+dalia/{s}/{s}.pkl', 'rb') as file:

        print(f'saving {s}')
        data = pickle.load(file, encoding='latin1')

        data_dict[s]['ppg'] = data['signal']['wrist']['BVP'][::2]     # downsample PPG to match fs_acc
        data_dict[s]['acc'] = data['signal']['wrist']['ACC']
        data_dict[s]['label'] = (data['label'])                       # ground truth EEG
        data_dict[s]['activity'] = data['activity']

        # alignment corrections
        data_dict[s]['ppg'] = data_dict[s]['ppg'][38:,:].T              # (1, n_samples)
        data_dict[s]['acc'] = data_dict[s]['acc'][:-38,:].T             # (3, n_samples)
        data_dict[s]['label'] = data_dict[s]['label'][:-1]              # (n_windows,)
        data_dict[s]['activity'] = data_dict[s]['activity'][:-1,:].T    # (1, n_samples)

        # window data
        data_dict = window_data(data_dict, s)

        # print(data_dict[s]['ppg'].shape)
        # print(data_dict[s]['acc'].shape)
        # print(data_dict[s]['label'].shape)
        # print(data_dict[s]['activity'].shape)

    return data_dict

def window_data(data_dict, s):
    '''
    Segment data into windows of 8 seconds with 2 second overlap
    :param data_dict: dictionary with all signals in arrays for given session
    :return: dictionary of windowed signals containing X and Y data
        ppg.shape = (n_windows, 1, 256)
        acc.shape = (n_windows, 3, 256)
        labels.shape = (n_windows,)
        activity.shape = (n_windows,)
    '''

    # sampling rates
    fs = {
        'ppg': 32,                  # fs_ppg = 64 in paper but downsampled to match acc
        'acc': 32,
        'activity': 4
    }

    n_windows = int(len(data_dict[s]['label']))

    for k, f in fs.items():
        # can alternatively use skimage.util.shape.view_as_windows method

        window = 8*f                        # size of window
        step = 2*f                          # size of step
        data = data_dict[s][k]

        if k == 'ppg':
            # (1, n_samples) -> (n_windows, 1, 256)
            data_dict[s][k] = np.zeros((n_windows, 1, window))
            for i in range(n_windows):
                start = i * step
                end = start + window
                data_dict[s][k][i, :, :] = data[:, start:end]

        if k == 'acc':
            data_dict[s][k] = np.zeros((n_windows, 3, window))
            for i in range(n_windows):
                start = i * step
                end = start + window
                data_dict[s][k][i, :, :] = data[:, start:end]

        if k == 'activity':
            data_dict[s][k] = np.zeros((n_windows,))
            for i in range(n_windows):
                start = i * step
                end = start + window
                data_dict[s][k][i] = data[0, start:end][0]             # take first value as value of whole window

    return data_dict

def normalise(X):
    '''
    Z-normalises data for all windows, across each channel
    :param X: of shape (4, 256, n_windows)
    :return:
        normalised X: of shape (4, 256, n_windows)
        ms (means): of shape (4, n_windows)
        stds (standard deviations) of shape (4, n_windows)
    '''

    ms = np.zeros((4, X.shape[-1]))          # mean of each (channel, window)
    stds = np.zeros((4, X.shape[-1]))        # stdev of each (channel, window)

    # iterate over channels
    for i in range(X.shape[0]):
        # create term to be updated, X_pres
        X_pres = X[i,:,:]

        # iterate over windows
        for j in range(X_pres.shape[-1]):

            m = np.mean(X_pres[:,j])
            std = np.std(X_pres[:,j])

            # Z-normalisation
            X_pres[:,j] = X_pres[:,j] - m

            if std != 0:
                X_pres[:,j] = X_pres[:,j] / std

            # save ms and stds
            ms[i, j] = m
            stds[i, j] = std

        # fill in X with updated X_pres
        X[i,:,:] = X_pres

    return X, ms, stds

def ma_removal(data_dict, sessions):
    '''
    Remove motion artifacts from raw PPG data by running through accelerometer_cnn
    :param data_dict: dictionary containing ppg, acc, label and activity data for each session
    :param s: list of sessions
    :return:
    '''

    X_dict = {}             # training data

    n_epochs = 500

    for s in sessions:

        # concatenate ppg + accelerometer signal data -> (n_windows, 4, 256)
        X = np.concatenate((data_dict[s]['ppg'], data_dict[s]['acc']), axis=1)
        print(X.shape)

        # find indices of activity changes
        idx = np.argwhere(np.abs(np.diff(data_dict[s]['activity'])) > 0).flatten() +1

        # add indices of start and end points
        idx = np.insert(idx, 0, 0)
        idx = np.insert(idx, idx.size, data_dict[s]['label'].shape[0])

        for i in tqdm(range(idx.size - 1)):

            X_pres = X[:,:,idx[i] : idx[i+1]]     # splice X into current activity

            # initialise CNN model
            model = AdaptiveLinearModel(n_epochs=n_epochs)
            sgd = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)
            model.local_optimizer = sgd

            # run training on single window
            model(X_pres)



def main():

    def save_dict(sessions, filename='ppg_dalia_dict'):

        # create dictionary to hold all data
        data_dict = {f'{session}': {} for session in sessions}

        # iterate over sessions
        for session in sessions:
            data_dict = save_data(session, data_dict)

        # save dictionary
        with open(filename, 'wb') as file:
            pickle.dump(data_dict, file)
        print(f'Data dictionary saved to {filename}')

        return

    def load_dict(filename='ppg_dalia_dict'):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # comment out save or load (to make a typed input switch)
    # save_dict(sessions)
    data_dict = load_dict()

    # pass accelerometer data through CNN
    ma_removal(data_dict, sessions)


if __name__ == '__main__':
    main()
