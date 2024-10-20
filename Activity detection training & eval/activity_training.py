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

# switch between models here
# from activity_model_tcn1 import AccModel
from activity_model_cnn2 import AccModel

def extract_activity(dict, sessions, mode):
    '''
    Remove transient regions from data - i.e. windows for which there is no activity label
    Take corresponding activity labels and signal data
    '''

    act_dict = {f'{session}': {} for session in sessions}

    for s in sessions:

        y = dict[s]['activity']
        # multiple channel input - create list containing each filtered variation of the sensor input
        X_ppg = [dict[s]['ppg'][j] for j in dict[s]['ppg'].keys()]
        X_acc = dict[s]['acc']

        # remove transient regions
        act_idx = np.where(y != 0)[0]
        y = y[act_idx]
        X_ppg = np.concatenate([X_ppg[i][act_idx] for i,_ in enumerate(X_ppg)], axis=1)
        X_acc = X_acc[act_idx]

        y = (y-1).astype(int)

        # create input tensors
        act_dict[s]['activity'] = y
        if mode == 'p':
            act_dict[s]['input'] = X_ppg
        elif mode == 'a':
            act_dict[s]['input'] = X_acc
        elif mode == 'x':
            act_dict[s]['input'] = np.concatenate((X_ppg, X_acc), axis=1)

        print(act_dict[s]['input'].shape)

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

        # plt.plot(fft_dict[s]['acc'][2000, 0, :])
        # plt.show()

        fft_dict[s]['activity'] = dict[s]['activity']

    return fft_dict

def z_normalise(dict, sessions):
    '''
    Z-normalises data for each window across each channel
    :param X: of shape (n_windows, n_channels, n_samples)
    :return X_norm: of shape (n_windows, n_channels, n_samples)
    '''

    for s in sessions:
        X = dict[s]['input']

        # calculate mean and stdev for each channel in each window - creates shape (n_windows, 4)
        ms = np.mean(X, axis=2)
        stds = np.std(X, axis=2)

        # reshape ms and stds to allow broadcasting
        ms_reshaped = ms[:, :, np.newaxis]
        stds_reshaped = stds[:, :, np.newaxis]

        # Z-normalisation
        X_norm = (X - ms_reshaped) / np.where(stds_reshaped != 0, stds_reshaped, 1)
        dict[s]['input'] = X_norm

    return dict

def train_model(dict, sessions, num_classes=8):

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = []
    y = []

    # create lists for training & label data
    x.extend([dict[session]['input'] for session in sessions])
    y.extend([dict[session]['activity'] for session in sessions])

    # initialise model
    n_epochs = 20
    batch_size = 128             # number of windows to be processed together
    n_splits = 4

    model = AccModel()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
    print(f"Number of trainable parameters: {count_parameters(model)}")

    # create batch splits
    ids = shuffle(list(range(len(sessions))))       # index each session
    splits = np.array_split(ids, n_splits)

    start_time = time.time()

    for split_idx, split in enumerate(splits):

        # set training data
        train_idxs = np.array([i for i in ids if i not in split])
        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)
        # X_train = np.expand_dims(X_train, axis=-2)                          # add height dimension to tensor

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
            # X_test = np.expand_dims(X_test, axis=-2)

            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)
            y_test = F.one_hot(y_test, num_classes=num_classes).float()

            # set validation data
            val_idxs = np.array([j for j in split if j != s])
            X_val = np.concatenate([x[j] for j in val_idxs], axis=0)
            y_val = np.concatenate([y[j] for j in val_idxs], axis=0)
            # X_val = np.expand_dims(X_val, axis=-2)

            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)
            y_val = F.one_hot(y_val, num_classes=num_classes).float()

            print(f'X_train shape: {X_train.shape}')
            print(f'y_train shape: {y_train.shape}')
            print(f'X_val shape: {X_val.shape}')
            print(f'y_val shape: {y_val.shape}')
            print(f'X_test shape: {X_test.shape}')
            print(f'y_test shape: {y_test.shape}')

            loss_func = nn.CrossEntropyLoss()
            train_losses = []
            val_losses = []

            # training loop
            for epoch in range(n_epochs):

                loss_epoch = 0.0
                model.train()

                # create training batches of windows to pass through model
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    ### input shape (batch_size, n_channels, n_samples)
                    ### output shape (batch_size, n_activities)

                    optimizer.zero_grad()
                    pred = model(X_batch)           # forward pass

                    # calculate training loss on distribution
                    loss = loss_func(pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()

                    print(f'Test session: S{s + 1}, Batch: [{batch_idx + 1}/{len(train_loader)}], '
                          f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {loss.item():.4f}')

                # track losses for plotting
                loss_epoch /= len(train_loader)
                train_losses.append(loss_epoch)

                # validation on whole validation set after each epoch
                model.eval()
                with torch.no_grad():
                    pred_val = model(X_val)
                    loss_val = loss_func(pred_val, y_val)
                    print(f'Test session: S{s + 1}, Epoch [{epoch + 1}/{n_epochs}], Validation Loss: {loss_val.item():.4f}')

                val_losses.append(loss_val.item())
                ### reinsert early stopping criteria here ###

            split_time = time.time()
            print("SINGLE SPLIT COMPLETE: time ", (split_time - start_time) / 3600, " hours.")

            plt.plot(train_losses, label='Train Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='black')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            # test on held-out session after all epochs complete
            with torch.no_grad():
                pred_test = model(X_test)
                loss_test = loss_func(pred_test, y_test)
                print(f'Test session: S{s + 1}, Test Loss: {loss_test.item():.4f}')

    end_time = time.time()
    print("TRAINING COMPLETE: time ", (end_time - start_time) / 3600, " hours.")


    return


def main():

    def load_dict(filename):

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(root_dir, filename)

        with open(filepath, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # load time series dictionary
    dict = load_dict(filename='ppg_dalia_dict_ppg_crm_1')

    # choose between ppg, acc or both as input
    mode = input("PPG (p), ACC (a) or both (x): ")
    if mode not in ['p', 'a', 'x']:
        print("Error: invalid input")

    # ignore transient periods (not assigned an activity)
    act_dict = extract_activity(dict, sessions, mode)

    # option to convert to frequency domain
    # act_dict = fourier(act_dict, sessions)

    # normalise
    act_dict = z_normalise(act_dict, sessions)

    # train model
    train_model(act_dict, sessions)




if __name__ == '__main__':
    main()


