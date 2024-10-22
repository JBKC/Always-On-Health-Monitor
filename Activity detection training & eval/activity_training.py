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
import training_analysis

# switch between models here
# from activity_model_tcn1 import AccModel
from activity_model_cnn2 import AccModel

def extract_activity(dict, sessions, mode):
    '''
    Remove transient regions from data - i.e. windows for which there is no activity label
    Take corresponding activity labels and signal data
    0 = sitting still
    1 = stairs
    2 = table football
    3 = cycling
    4 = driving
    5 = lunch break
    6 = walking
    7  = working at desk
    '''

    def plot_inputs(channels, label):
        '''
        Plot arbitrary period within input signals to compare filtering effects for a given activity label
        :param channels: contains the signals in shape (n_windows, n_channels, n_samples)
        :param labels: array of y axis labels of length (n_windows)
        '''

        # add labels of interest
        act_labels = [0, 3, 4]
        # list of arrays containing indices of activities
        act_idxs = [np.where(label == i)[0] for i in act_labels]

        fig, axs = plt.subplots(nrows=channels.shape[1], ncols=len(act_idxs), figsize=(10, 7))

        for i, channel in enumerate(range(channels.shape[1])):  # Iterate over channels (rows)
            for j, idxs in enumerate(act_idxs):  # Iterate over activity indices (columns)
                if len(idxs) > 0:
                    # plot arbitrary 8-second window from each activity
                    axs[i, j].plot(channels[idxs[10], channel, :])
                    axs[i, j].set_title(f'Channel {i}, Activity {act_labels[j]}')

        plt.tight_layout()
        plt.show()

        return

    def plot_fft(channels, label, fs=32, cutoff=2):
        '''
        Plot averaged frequency domain plots for each activity
        :param channels: contains the accelerometer data in shape (n_windows, n_channels, n_samples)
        :param labels: array of y axis labels of length (n_windows)
        '''


        n_channels = channels.shape[1]
        n_samples = channels.shape[-1]
        act_labels = np.unique(label)

        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 7))

        # iterate over activities
        for i, activity in enumerate(act_labels):
            act_idxs = np.where(label == activity)[0]

            avg_ffts = np.zeros((n_channels, n_samples // 2))

            # iterate over accelerometer channels
            for channel in range(n_channels):
                ffts = []
                for idx in act_idxs:
                    # take FFT of each window
                    fft = np.fft.fft(channels[idx, channel, :])
                    fft = np.abs(fft)[:n_samples // 2]
                    ffts.append(fft)

                # compute mean FFT across all windows for each channel
                avg_ffts[channel] = np.mean(ffts, axis=0)

            # get frequencies
            freqs = np.fft.fftfreq(n_samples, 1 / fs)[:n_samples // 2]
            keep_idx = np.where(freqs > cutoff)
            freqs = freqs[keep_idx]

            ax = axs[i // 4, i % 4]
            for channel in range(n_channels):
                ax.plot(freqs, avg_ffts[channel][keep_idx], label=f'Channel {channel}')

            ax.set_title(f'Activity {activity}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

        return

    act_dict = {f'{session}': {} for session in sessions}

    for s in sessions:

        y = dict[s]['activity']
        # multiple channel input - create list containing each filtered variation of the sensor input
        X_ppg = [dict[s]['ppg'][j] for j in list(dict[s]['ppg'].keys())[1:]]         # ignore unfiltered (og) PPG signal
        X_acc = dict[s]['acc']

        # remove transient regions
        act_idx = np.where(y != 0)
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

        in_channels = act_dict[s]['input'].shape[1]

        # plotting options
        # if mode == 'p':
        #     plot_inputs(channels=act_dict[s]['input'], label=act_dict[s]['activity'])
        # if mode == 'a':
        #     plot_fft(channels=act_dict[s]['input'], label=act_dict[s]['activity'])

    return act_dict, in_channels

def fourier(dict, sessions):
    '''
    Create frequency-domain dictionary
    :param dict: time series dictionary
    :param sessions: list of sessions
    :return fft_dict: frequency domain dictionary
    '''

    fft_dict = {f'{session}': {} for session in sessions}

    for s in sessions:

        X = dict[s]['input']
        fft_acc = np.abs(np.fft.fft(X, axis=-1))
        # take relevant part of FFT
        fft_acc = fft_acc[:,:,1:X.shape[-1]//2+1]

        # plt.plot(fft_acc[10, 0, :])
        # plt.show()

        # update dictionary
        fft_dict[s]['input'] = fft_acc
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

def train_model(dict, sessions, in_channels, num_classes=8):

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = []
    y = []

    # create lists for training & label data
    x.extend([dict[session]['input'] for session in sessions])
    y.extend([dict[session]['activity'] for session in sessions])

    # initialise model
    n_epochs = 1
    batch_size = 128             # number of windows to be processed together
    n_splits = 4

    model = AccModel(in_channels, num_classes)
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

            hook = training_analysis.register_hook(model.initial_block)         # forward hook for activation maps

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

                ### training analysis - plot activations at the end of each epoch
                activation_map = training_analysis.activations[model.initial_block]
                training_analysis.plot_activation_map(activation_map)

            hook.remove()

            split_time = time.time()
            print("SINGLE SPLIT COMPLETE: time ", (split_time - start_time) / 3600, " hours.")


            plt.plot(train_losses, label='Train Loss', color='blue')
            plt.plot(val_losses, label='Validation Loss', color='black')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            ### training analysis - plot weight distributions (single split)
            training_analysis.plot_weight_dist(model)

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
    dict = load_dict(filename='ppg_dalia_dict_ppg_crm_v1')

    # choose between ppg, acc or all as input
    mode = input("PPG (p), ACC (a) or all (x): ")
    if mode not in ['p', 'a', 'x']:
        print("Error: invalid input")

    # ignore transient periods (not assigned an activity) to assign labels
    act_dict, in_channels = extract_activity(dict, sessions, mode)

    # option to convert to frequency domain
    # act_dict = fourier(act_dict, sessions)

    # normalise
    act_dict = z_normalise(act_dict, sessions)

    # train model
    train_model(act_dict, sessions, in_channels)




if __name__ == '__main__':
    main()


