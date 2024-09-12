'''
Main script for training temporal attention model
'''

import numpy as np
import pickle
from sklearn.utils import shuffle
import time
import pandas as pd


def temporal_pairs(dict, split):
    '''
    Create temporal pairs between adjacent windows for training/testing splits
    :param
        dict: dictionary of session data - each session shape (n_windows, n_channels, n_samples)
        split: list of session names in split
    :return: list of temporal pairs (x_split), labels (y_split) and activities (act_split)
    '''

    x_split = []
    y_split = []
    act_split = []

    for s in split:

        x = dict[s]['bvp']

        # pair adjacent windows (i, i+1)
        x_pairs = (np.expand_dims(x[:-1,:],axis=-1) , np.expand_dims(x[1:,:],axis=-1))
        x_pairs = np.concatenate(x_pairs,axis=-1)
        # results in concatenated pairs of shape (n_windows, 1, n_samples, 2)
        x_split.append(x_pairs)

        y_split.append(dict[s]['label'][1:])
        act_split.append(dict[s]['activity'][1:])

    return x_split, y_split, act_split

def train_model(dict, sessions):
    '''
    Create Leave One Session Out split
    :param dict:
    :param sessions:
    :return:
    '''

    n_epochs = 500
    batch_size = 256
    n_splits = 4

    # LOSO splits
    ids = shuffle(sessions)
    splits = np.array_split(ids, n_splits)

    # train model
    start_time = time.time()

    for split in splits:

        # create temporal pairs of time windows
        x, y, act = temporal_pairs(dict, split)


        print(test_idxs)


def main():

    def load_dict(filename='ppg_filt_dict'):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # load dictionary
    dict = load_dict()
    train_model(dict, sessions)





if __name__ == '__main__':
    main()

