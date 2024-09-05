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
from torch.utils.data import DataLoader, TensorDataset


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

def z_normalise(X):
    '''
    Z-normalises data for all windows, across each channel
    :param X: of shape (n_windows, 4, 256)
    :return:
        normalised X: of shape (n_windows, 4, 256)
    '''

    ms = np.zeros((X.shape[0], 4))          # mean of each (window, channel)
    stds = np.zeros((X.shape[0], 4))        # stdev of each (window, channel)

    # iterate over channels
    for j in range(X.shape[1]):
        # create term to be updated, X_pres
        X_pres = X[:,j,:]

        # iterate over windows
        for i in range(X_pres.shape[0]):

            m = np.mean(X_pres[i,:])
            std = np.std(X_pres[i,:])

            # Z-normalisation
            X_pres[i,:] = X_pres[i,:] - m
            if std != 0:
                X_pres[i,:] = X_pres[i,:] / std

            # save ms and stds
            ms[i, j] = m
            stds[i, j] = std

        # fill in X with updated X_pres
        X[:,j,:] = X_pres

    return X, ms, stds

def undo_normalisation(X, ms, stds):
    '''
    Transform cleaned PPG signal back into original space following filtering
    :param X: of shape (n_windows, 1, 256)
    :return:
        normalised X: of shape (n_windows, 1, 256)
        ms (means): of shape (n_windows, 4)
        stds (standard deviations) of shape (n_windows, 4)
    '''

    # iterate over channels (only ppg channel now)
    for j in range(X.shape[1]):
        # create term to be updated, X_pres
        X_pres = X[:,j,:]

        # iterate over windows
        for i in range(X_pres.shape[0]):
            # reverse normalisation
            if stds[i,j] != 0:
                X_pres[i, :] = X_pres[i, :] * stds[i,j]
            X_pres[i, :] = X_pres[i, :] + ms[i,j]

        # fill in X with updated X_pres
        X[:,j,:] = X_pres

    return X


def ma_removal(data_dict, sessions):
    '''
    Remove session-specific motion artifacts from raw PPG data by training on accelerometer_cnn
    :param data_dict: dictionary containing ppg, acc, label and activity data for each session
    :param s: list of sessions
    :return:
    '''

    X_BVP = []          # filtered PPG data

    # initialise CNN model
    n_epochs = 16000
    model = AdaptiveLinearModel(n_epochs=n_epochs)
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)

    # for s in sessions:
    for s in sessions[:1]:
        # concatenate ppg + accelerometer signal data -> (n_windows, 4, 256)
        X = np.concatenate((data_dict[s]['ppg'], data_dict[s]['acc']), axis=1)

        # find indices of activity changes (marks batches)
        idx = np.argwhere(np.abs(np.diff(data_dict[s]['activity'])) > 0).flatten() +1

        # add indices of start and end points
        idx = np.insert(idx, 0, 0)
        idx = np.insert(idx, idx.size, data_dict[s]['label'].shape[0])

        # create batches
        # for i in range(idx.size - 1):
        for i in range(1):
            # (batch_size, channels, height, width) = (batch_size, 1, 3, 256)
            X_pres = X[idx[i] : idx[i+1],:,:]           # splice X into current activity

            # batch Z-normalisation
            X_pres, ms, stds = z_normalise(X_pres)

            X_pres = np.expand_dims(X_pres, axis=1)     # add channel dimension
            X_pres = torch.from_numpy(X_pres).float()

            # run training loop
            x_out = model.train_batch(X=X_pres, session=s, batch=i, n_epochs=n_epochs, optimizer=optimizer)

            # denormalise
            x_out = undo_normalisation(x_out, ms, stds)

            # append filtered batch
            X_BVP.append(x_out)

        X_BVP = np.concatenate(X_BVP, axis=0)

        X_BVP = X_BVP.flatten()
        X_PPG = data_dict[s]['ppg'].flatten()

        # unravel & test plot
        print(X_BVP.shape)
        print(X_PPG.shape)

        plt.plot(X_PPG)
        plt.plot(X_BVP)
        plt.show()


    return X_BVP


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
