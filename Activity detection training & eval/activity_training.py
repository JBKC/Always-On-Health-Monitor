'''
Train model on PPG and accelerometer data to detect activity
'''

import numpy as np
import pickle
from sklearn.utils import shuffle
import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from activity_model import AccModel


def fourier(dict, sessions):
    '''
    Create frequency-domain dictionary for accelerometer only
    :param dict: time series dictionary
    :param sessions: list of sessions
    :return fft_dict: frequency domain dictionary
    '''

    fft_dict = {f'{session}': {} for session in sessions}

    for s in sessions:

        X = dict[s]['acc']
        fft_acc = np.abs(np.fft.fft(X, axis=1))
        fft_dict[s]['acc'] = fft_acc

        fft_dict[s]['activity'] = dict[s]['activity']

    print(fft_dict['S3']['activity'].shape)
    print(dict['S3']['activity'].shape)
    print(fft_dict['S14']['acc'].shape)
    print(dict['S14']['acc'].shape)

    return fft_dict



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

    return X_norm



def train_model(dict, sessions):

    x = []
    y = []

    # create lists for training & label data
    x.extend([dict[session]['acc'] for session in sessions])
    y.extend([dict[session]['activity'] for session in sessions])

    # initialise model
    n_epochs = 500
    patience = 10               # early stopping parameter
    batch_size = 256            # number of windows to be processed together
    n_splits = 4

    # model = AccModel()
    # optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)

    # create batch splits
    ids = shuffle(list(range(len(sessions))))       # index each session
    splits = np.array_split(ids, n_splits)

    start_time = time.time()

    for split_idx, split in enumerate(splits):

        # set training data
        train_idxs = np.array([i for i in ids if i not in split])
        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # create TensorDataset and DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for session_idx, s in enumerate(split):

            # set current session to test data
            X_test = x[s]
            y_test = y[s]

            print(y_test)
            print(y_test.shape)
            print(X_test.shape)

            # set validation data
            val_idxs = np.array([j for j in split if j != s])
            X_val = np.concatenate([x[j] for j in val_idxs], axis=0)
            y_val = np.concatenate([y[j] for j in val_idxs], axis=0)

            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)


            # training loop
            for epoch in range(epoch +1, n_epochs):

                model.train()



    return


def main():

    def load_dict(filename='ppg_dalia_dict'):

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(root_dir, filename)

        with open(filepath, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # load time series dictionary
    dict = load_dict()

    # convert to frequency domain & normalise
    dict = z_normalise(fourier(dict, sessions))

    train_model(dict, sessions)




if __name__ == '__main__':
    main()


