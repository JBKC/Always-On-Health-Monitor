'''
Train model on PPG and accelerometer data to detect activity
'''

import numpy as np
import pickle
from sklearn.utils import shuffle
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from activity_model_cnn1 import AccModel


def extract_activity(dict, sessions):
    '''
    Remove transient regions from data - i.e. where there is no activity label
    '''

    act_dict = {f'{session}': {} for session in sessions}

    for s in sessions:

        y = dict[s]['activity']
        X = dict[s]['acc']

        # remove transient regions
        act_idx = np.where(y != 0)[0]
        y = y[act_idx]
        X = X[act_idx]
        y = (y-1).astype(int)

        # add to dictionary
        act_dict[s]['activity'] = y
        act_dict[s]['acc'] = X

    return act_dict

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
        fft_acc = np.abs(np.fft.fft(X, axis=-1))
        # take relevant part of FFT
        fft_acc = fft_acc[:,:,1:X.shape[-1]//2+1]

        # normalise
        fft_dict[s]['acc'] = z_normalise(fft_acc)

        # plt.plot(fft_dict[s]['acc'][2000, 0, :])
        # plt.show()

        fft_dict[s]['activity'] = dict[s]['activity']

    return fft_dict

def z_normalise(X):
    '''
    Z-normalises data for each window across each channel
    :param X: of shape (n_windows, 3, n_fft)
    :return X_norm: of shape (n_windows, 3, n_fft)
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

def train_model(dict, sessions, num_classes=8):

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

    model = AccModel()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)

    # create batch splits
    ids = shuffle(list(range(len(sessions))))       # index each session
    splits = np.array_split(ids, n_splits)

    start_time = time.time()

    for split_idx, split in enumerate(splits):

        # set training data
        train_idxs = np.array([i for i in ids if i not in split])
        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)
        X_train = np.expand_dims(X_train, axis=-2)                          # add height dimension to tensor

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_train = F.one_hot(y_train, num_classes=num_classes).float()       # one-hot encode labels

        # create TensorDataset and DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for session_idx, s in enumerate(split):

            # set current session to test data
            X_test = x[s]
            y_test = y[s]
            X_test = np.expand_dims(X_test, axis=-2)

            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)
            y_test = F.one_hot(y_test, num_classes=num_classes).float()

            # set validation data
            val_idxs = np.array([j for j in split if j != s])
            X_val = np.concatenate([x[j] for j in val_idxs], axis=0)
            y_val = np.concatenate([y[j] for j in val_idxs], axis=0)
            X_val = np.expand_dims(X_val, axis=-2)

            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)
            y_val = F.one_hot(y_val, num_classes=num_classes).float()

            # print(X_train.shape)
            # print(X_val.shape)
            # print(X_test.shape)
            #
            # print(y_train.shape)
            # print(y_val.shape)
            # print(y_test.shape)

            loss_func = nn.CrossEntropyLoss()

            # training loop
            for epoch in range(n_epochs):

                model.train()

                # create training batches of windows to pass through model
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    ### input shape (batch_size, n_channels, n_fft, 1) = (256, 3, 128, 1)
                    ### output shape (batch_size, num_classes)

                    optimizer.zero_grad()
                    pred = model(X_batch)           # forward pass

                    # calculate training loss on distribution
                    loss = loss_func(pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    print(f'Test session: S{s + 1}, Batch: [{batch_idx + 1}/{len(train_loader)}], '
                          f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {loss.item():.4f}')

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

    # ignore transient periods (not assigned an activity)
    act_dict = extract_activity(dict, sessions)

    # convert to frequency domain & normalise
    f_dict = fourier(act_dict, sessions)

    # train model
    train_model(f_dict, sessions)




if __name__ == '__main__':
    main()


