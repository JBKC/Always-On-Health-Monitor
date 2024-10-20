'''
Pull, filter and experiment with feature extraction for activity detection
'''

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import butter, filtfilt


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

    def butter_filter(signal, btype, lowcut=None, highcut=None, fs=32, order=5):
        """
        Applies Butterworth filter
        :param signal: input signal of shape (n_channels, n_samples)
        :return smoothed: smoothed signal of shape (n_channels, n_samples)
        """

        nyquist = 0.5 * fs

        if btype == 'bandpass':
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype=btype)
        elif btype == 'lowpass':
            high = highcut / nyquist
            b, a = butter(order, high, btype=btype)
        elif btype == 'highpass':
            low = lowcut / nyquist
            b, a = butter(order, low, btype=btype)

        # apply filter using filtfilt (zero-phase filtering)
        filtered = np.array([filtfilt(b, a, channel) for channel in signal])

        return filtered

    def plot_inputs(signals,fs=32,start=3000,period=5):
        '''
        Plot arbitrary period within input signals to compare filtering effects
        :param signals: list of signals to plot
        :param start time (s) - max c.75000
        :param period (s)
        '''

        start = start * fs
        period = period * fs

        fig, axs = plt.subplots(4, 1, figsize=(8, 8))
        for i, ax in enumerate(axs):
            ax.plot(signals[i][0,start:start+period])
        plt.tight_layout()
        plt.show()

        return


    with open(f'{root_dir}/ppg+dalia/{s}/{s}.pkl', 'rb') as file:

        print(f'processing {s}')
        data = pickle.load(file, encoding='latin1')

        # get raw data from pkl file
        data_dict[s]['ppg'] = data['signal']['wrist']['BVP'][::2]     # downsample PPG to match fs_acc
        data_dict[s]['acc'] = data['signal']['wrist']['ACC']
        data_dict[s]['label'] = (data['label'])                       # ground truth EEG
        data_dict[s]['activity'] = data['activity']
        # alignment corrections
        data_dict[s]['ppg'] = data_dict[s]['ppg'][38:,:].T              # (1, n_samples)
        data_dict[s]['acc'] = data_dict[s]['acc'][:-38,:].T             # (3, n_samples)
        data_dict[s]['label'] = data_dict[s]['label'][:-1]              # (n_windows,)
        data_dict[s]['activity'] = data_dict[s]['activity'][:-1,:].T    # (1, n_samples)

        # applying filtering ranges to PPG
        data_dict[s]['ppg_c'] = butter_filter(signal=data_dict[s]['ppg'],btype='bandpass',lowcut=0.5,highcut=4)       # cardiac
        data_dict[s]['ppg_r'] = butter_filter(signal=data_dict[s]['ppg'],btype='bandpass',lowcut=0.2,highcut=0.35)    # respiratory
        data_dict[s]['ppg_m'] = butter_filter(signal=data_dict[s]['ppg'],btype='highpass',lowcut=0.1)                 # motion artifacts

        print(data_dict[s]['ppg_c'].shape)

        # plot
        plot_inputs([ data_dict[s]['ppg'],data_dict[s]['ppg_c'],data_dict[s]['ppg_r'],data_dict[s]['ppg_m'] ])


        # data_dict[s]['acc'] = butter_filter(signal=data_dict[s]['acc'])


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


    sessions = [f'S{i}' for i in range(1, 16)]


    save_dict(sessions, "ppg_dalia_dict_f_0.3-10")



if __name__ == '__main__':
    main()
