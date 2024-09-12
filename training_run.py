'''
Main script for training temporal attention model
'''

import numpy as np
import pickle
from sklearn.utils import shuffle
import time
from temporal_attention_model import TemporalAttentionModel
import torch.optim as optim


def temporal_pairs(dict, split):
    '''
    Create temporal pairs between adjacent windows for all data
    :param
        dict: dictionary of session data - each session shape (n_windows, n_channels, n_samples)
        split: list of session names in split
    :return: lists of temporal pairs (x_split), labels (y_split) and activities (act_split)
    '''

    x_all = []
    y_all = []
    act_all = []

    for s in split:

        x = dict[s]['bvp']

        # pair adjacent windows (i, i+1)
        x_pairs = (np.expand_dims(x[:-1,:],axis=-1) , np.expand_dims(x[1:,:],axis=-1))
        x_pairs = np.concatenate(x_pairs,axis=-1)
        # results in concatenated pairs of shape (n_windows, 1, n_samples, 2)
        x_all.append(x_pairs)

        y_all.append(dict[s]['label'][1:])
        act_all.append(dict[s]['activity'][1:])

    return x_all, y_all, act_all

def train_model(dict, sessions):
    '''
    Create Leave One Session Out split and run through model
    :param dict:
    :param sessions:
    :return:
    '''

    # initialise model
    n_epochs = 500
    batch_size = 256
    n_splits = 4
    model = TemporalAttentionModel(n_epochs=n_epochs)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)

    # create temporal pairs of time windows
    x, y, act = temporal_pairs(dict, sessions)

    # LOSO splits
    ids = shuffle(list(range(15)))                  # index each session
    splits = np.array_split(ids, n_splits)

    start_time = time.time()
    for split in splits:

        # set training data (current split = testing/validation data)
        train_idxs = np.array([i for i in ids if i not in split])
        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)
        act_train = np.concatenate([act[i] for i in train_idxs], axis=0)

        # create inner LOSO split to get testing & validation split
        for s in split:

            # set current session to test data
            X_test = x[s]
            y_test = y[s]
            act_test = act[s]

            # set validation data
            val_idxs = np.array([j for j in split if j != s])
            X_val = np.concatenate([x[j] for j in val_idxs], axis=0)
            y_val = np.concatenate([y[j] for j in val_idxs], axis=0)
            act_val = np.concatenate([act[j] for j in val_idxs], axis=0)



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

