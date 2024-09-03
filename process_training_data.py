'''
Initial file for pulling and processing training data from PPG-DaLiA dataset
'''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows

def save_data(s, data_dict):
    '''
    Pull raw data from PPG Dalia files and save down to a dictionary
    :param s: session name
    :param data_dict: empty dictionary to hold data
    :return:
    '''

    def unpack(data):
        # data = data.reshape(-1, 1)
        data = np.squeeze(data)
        return data

    # pull raw data
    with open(f'ppg+dalia/{s}/{s}.pkl', 'rb') as file:

        print(f'saving {s}')
        data = pickle.load(file, encoding='latin1')

        data_dict[s]['ppg'] = unpack(data['signal']['wrist']['BVP'][::2])     # downsample PPG to match fs_acc
        data_dict[s]['acc'] = unpack(data['signal']['wrist']['ACC'])
        data_dict[s]['label'] = unpack(data['label'])        # ground truth EEG
        data_dict[s]['activity'] = data['activity']

        # alignment corrections
        data_dict[s]['ppg'] = data_dict[s]['ppg'][38:, ...]
        data_dict[s]['acc'] = data_dict[s]['acc'][:-38, ...]
        data_dict[s]['label'] = data_dict[s]['label'][:-1]
        data_dict[s]['activity'] = data_dict[s]['activity'][:-1]

        # window data
        data_dict = window_data(data_dict, s)

        # print(data_dict['S1']['ppg'].shape)
        # print(data_dict['S1']['acc'].shape)
        # print(data_dict['S1']['label'].shape)
        # print(data_dict['S1']['activity'].shape)

    return data_dict

def window_data(data_dict, s):
    '''
    Segment data into windows of 8 seconds with 2 second overlap
    :param data_dict: dictionary with all signals in arrays for given session
    :return: dictionary of windowed signals containing X and Y data
        ppg.shape = (window length, n_windows)
        acc.shape = (3, 256, n_windows) - to match Pytorch format
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
            data_dict[s][k] = np.zeros((window, n_windows))
            for i in range(n_windows):
                start = i * step
                end = start + window
                data_dict[s][k][:, i] = data[start:end]

        if k == 'acc':
            data_dict[s][k] = np.zeros((data.shape[-1], window, n_windows))
            for i in range(n_windows):
                start = i * step
                end = start + window
                data_dict[s][k][:, :, i] = data[start:end, :].T

        if k == 'activity':
            data_dict[s][k] = np.zeros((n_windows,))
            for i in range(n_windows):
                start = i * step
                end = start + window
                data_dict[s][k][i] = data[start:end][0]             # take first value as value of whole window

    return data_dict

def ma_removal(data_dict):
    '''
    Remove motion artifacts from raw PPG data by running through
    :param data_dict: dictionary containing all X and y data
    :return:
    '''


    return

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
    ma_removal(data_dict)


if __name__ == '__main__':
    main()
