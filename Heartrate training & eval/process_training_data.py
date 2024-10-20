'''
Initial file for pulling and processing training data from PPG-DaLiA dataset
'''

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from accelerometer_cnn import AdaptiveLinearModel
from scipy.signal import butter, filtfilt
import generate_adversarial_dataset


def save_data(s, data_dict, root_dir, filename):
    '''
    Pull raw data from PPG Dalia files and save down to a dictionary
    :param s: session name
    :param data_dict: empty dictionary to hold data
    :return: data_dict: filled dictionary with all PPG Dalia data
    '''

    def mean_smooth(signal, window_size=8):
        """
        Applies mean smoothing filter
        :param signal: input signal of shape (n_channels, n_samples)
        :param window_size: size of window over which to apply smoothing
        :return smoothed: smoothed signal of shape (n_channels, n_samples)
        """

        # smoothing kernel
        kernel = np.ones(window_size) / window_size

        # Apply convolution along the last dimension for each channel
        smoothed = np.array([np.convolve(channel, kernel, mode='same') for channel in signal])
        # plt.plot(signal[0,:])
        # plt.plot(smoothed[0,:])
        # plt.show()

        return smoothed

    def butter_filter(signal, lowcut=0.1, highcut=10, fs=32, order=4):
        """
        Applies Butterworth filter
        :param signal: input signal of shape (n_channels, n_samples)
        :return smoothed: smoothed signal of shape (n_channels, n_samples)
        """

        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist

        # Create the Butterworth bandpass filter
        b, a = butter(order, [low, high], btype='bandpass')

        # Apply the filter to the signal using filtfilt (zero-phase filtering)
        filtered = np.array([filtfilt(b, a, channel) for channel in signal])

        return filtered

    # pull raw data
    with open(f'{root_dir}/ppg+dalia/{s}/{s}.pkl', 'rb') as file:

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

        # filter all the inputs - currently used for activity detection
        if filename == "ppg_dalia_dict_filtered":
            print(data_dict[s]['ppg'].shape)

            # plt.plot(data_dict[s]['acc'][1,:])
            data_dict[s]['ppg'] = butter_filter(signal=data_dict[s]['ppg'])
            data_dict[s]['acc'] = butter_filter(signal=data_dict[s]['acc'])
            # plt.plot(data_dict[s]['acc'][1,:])
            # plt.show()

        # window data
        data_dict = window_data(data_dict, s)

        print(data_dict[s]['ppg'].shape)
        print(data_dict[s]['acc'].shape)
        print(data_dict[s]['label'].shape)
        print(data_dict[s]['activity'].shape)

    return data_dict

def window_data(data_dict, s):
    '''
    Segment data into windows of 8 seconds with 2 second overlap. Only used when saving down raw data for first time
    :param data_dict: dictionary with all signals in arrays for given session
    :return: dictionary of windowed signals containing X and Y data
        ppg.shape = (n_windows, 1, 256)
        acc.shape = (n_windows, 3, 256)
        labels.shape = (n_windows,)
        activity.shape = (n_windows,)
    '''

    # sampling rates
    fs = {
        'ppg': 32,                  # fs_ppg = 64 in paper but downsampled to match accelerometer
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
    Z-normalises data for all windows, across each channel, using vectorisation
    :param X: of shape (n_windows, 4, 256)
    :return:
        X_norm: of shape (n_windows, 4, 256)
        ms (means): of shape (n_windows, 4)
        stds (standard deviations) of shape (n_windows, 4)
    '''

    # calculate mean and stdev for each channel in each window - creates shape (n_windows, 4)
    ms = np.mean(X, axis=2)
    stds = np.std(X, axis=2)

    # reshape ms and stds to allow broadcasting
    ms_reshaped = ms[:, :, np.newaxis]
    stds_reshaped = stds[:, :, np.newaxis]

    # Z-normalisation
    X_norm = (X - ms_reshaped) / np.where(stds_reshaped != 0, stds_reshaped, 1)

    return X_norm, ms, stds

def undo_normalisation(X_norm, ms, stds):
    '''
    Transform cleaned PPG signal back into original space following filtering
    :params:
        X_norm: of shape (n_windows, 1, 256)
        ms (means): of shape (n_windows, 4)
        stds (standard deviations) of shape (n_windows, 4)
    :return:
        X: of shape (n_windows, 1, 256)
    '''

    ms_reshaped = ms[:, :, np.newaxis]
    stds_reshaped = stds[:, :, np.newaxis]

    return (X_norm * np.where(stds_reshaped != 0, stds_reshaped, 1)) + ms_reshaped

def ma_removal(data_dict, sessions):
    '''
    Remove session-specific motion artifacts from raw PPG data by training on accelerometer_cnn
    Save down to dictionary "ppg_filt_dict"
    :param data_dict: dictionary containing ppg, acc, label and activity data for each session
    :param s: list of sessions
    :return: ppg_filt_dict: dictionary containing bvp (cleaned ppg) along with acc, label and activity
    '''

    # ppg_dalia_dict filtered for motion artifacts

    ppg_filt_dict = {f'{session}': {} for session in sessions}

    # initialise CNN model
    n_epochs = 1000
    model = AdaptiveLinearModel()
    optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=1e-2)

    for s in sessions:

        X_BVP = []  # filtered PPG data

        # concatenate ppg + accelerometer signal data -> (n_windows, 4, 256)
        X = np.concatenate((data_dict[s]['ppg'], data_dict[s]['acc']), axis=1)

        # find indices of activity changes (marks batches)
        idx = np.argwhere(np.abs(np.diff(data_dict[s]['activity'])) > 0).flatten() +1

        # add indices of start and end points
        idx = np.insert(idx, 0, 0)
        idx = np.insert(idx, idx.size, data_dict[s]['label'].shape[0])

        initial_state = model.state_dict()

        # iterate over activities (batches)
        for i in range(idx.size - 1):

            model.load_state_dict(initial_state)

            # create batches
            X_batch = X[idx[i]: idx[i + 1], :, :]  # splice X into current activity

            # batch Z-normalisation
            X_batch, ms, stds = z_normalise(X_batch)

            X_batch = np.expand_dims(X_batch, axis=1)          # add channel dimension
            X_batch = torch.from_numpy(X_batch).float()

            # accelerometer data are inputs:
            x_acc = X_batch[:, :, 1:, :]                 # (batch_size, 1, 3, 256)
            # PPG data are targets:
            x_ppg = X_batch[:, :, :1, :]                 # (batch_size, 1, 1, 256)

            # training loop
            for epoch in range(n_epochs):

                # forward pass through CNN to get x_ma (motion artifact estimate)
                x_ma = model(x_acc)
                # compute loss against raw PPG data
                loss = model.adaptive_loss(y_true=x_ppg, y_pred=x_ma)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Session {s}, Batch: [{i + 1}/{idx.size - 1}], '
                      f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

            # subtract the motion artifact estimate from raw signal to extract cleaned BVP
            with torch.no_grad():
                x_bvp = x_ppg[:, 0, 0, :] - x_ma

            # get signal into original shape: (n_windows, 1, 256)
            x_bvp = torch.unsqueeze(x_bvp, dim=1).numpy()

            # denormalise
            x_bvp = undo_normalisation(x_bvp, ms, stds)
            # keep only BVP (remove ACC)
            x_bvp = np.expand_dims(x_bvp[:,0,:], axis=1)

            # append filtered batch
            X_BVP.append(x_bvp)

        # add to dictionary
        X_BVP = np.concatenate(X_BVP, axis=0)

        ppg_filt_dict[s]['bvp'] = X_BVP
        ppg_filt_dict[s]['acc'] = data_dict[s]['acc']
        ppg_filt_dict[s]['label'] = data_dict[s]['label']
        ppg_filt_dict[s]['activity'] = data_dict[s]['activity']

        print(f"{s} shape: {ppg_filt_dict[s]['bvp'].shape}")

    # save dictionary
    with open('ppg_filt_dict', 'wb') as file:
        pickle.dump(ppg_filt_dict, file)
    print(f'Data dictionary saved to ppg_filt_dict')

    return

def main():

    def save_dict(sessions, filename='ppg_dalia_dict'):

        # create dictionary to hold all data
        data_dict = {f'{session}': {} for session in sessions}

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(root_dir, filename)

        # iterate over sessions
        for session in sessions:
            data_dict = save_data(session, data_dict, root_dir, filename)

        # save dictionary

        with open(filepath, 'wb') as file:
            pickle.dump(data_dict, file)
        print(f'Data dictionary saved to {filename}')

        return

    def load_dict(filename='ppg_dalia_dict'):

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(root_dir, filename)

        with open(filepath, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # comment out save or load
    save_dict(sessions, "ppg_dalia_dict_filtered")
    # data_dict = load_dict()

    # pass accelerometer data through CNN & save down new filtered data
    # ma_removal(data_dict, sessions)


if __name__ == '__main__':
    main()
