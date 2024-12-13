'''
Train model on PPG and accelerometer data to detect activity
Condensed version with the essentials
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
from activity_model_tcn2 import AccModel            # PPG-NeXt
from activity_model_tcn2 import AccModel            # ModernTCN

def extract_activity(dict, sessions):
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

    act_dict = {f'{session}': {} for session in sessions}

    for s in sessions:

        y = dict[s]['activity']
        # multiple channel input - create list containing each filtered variation of the sensor input
        X_ppg = [dict[s]['ppg'][j] for j in list(dict[s]['ppg'].keys())[1:]]         # ignore unfiltered (og) PPG signal - take the filtered variations
        X_acc = dict[s]['acc']

        # remove transient regions
        act_idx = np.where(y != 0)
        y = y[act_idx]
        X_ppg = np.concatenate([X_ppg[i][act_idx] for i,_ in enumerate(X_ppg)], axis=1)
        X_acc = X_acc[act_idx]

        y = (y-1).astype(int)

        # create input tensors
        act_dict[s]['activity'] = y
        act_dict[s]['input'] = np.concatenate((X_ppg, X_acc), axis=1)

        in_channels = act_dict[s]['input'].shape[1]

    return act_dict, in_channels


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

    def label_smooth(labels, smoothing, num_classes):
        with torch.no_grad():
            labels = labels * (1 - smoothing) + smoothing / num_classes
        return labels


    x = []
    y = []

    # create lists for training & label data
    x.extend([dict[session]['input'] for session in sessions])
    y.extend([dict[session]['activity'] for session in sessions])

    # initialise model
    n_epochs = 500
    patience = 10               # early stopping parameter
    batch_size = 128            # number of windows to be processed together
    n_splits = 4
    l2_lambda = 0.01            # regularisation

    # create batch splits
    ids = shuffle(list(range(len(sessions))))       # index each session
    splits = np.array_split(ids, n_splits)

    start_time = time.time()

    # outer LOSO split for training data
    for split_idx, split in enumerate(splits):

        # set training data
        train_idxs = np.array([i for i in ids if i not in split])
        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)
        # X_train = np.expand_dims(X_train, axis=-2)                          # add height dimension to tensor

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_train = F.one_hot(y_train, num_classes=num_classes).float()       # one-hot encode labels
        # apply label smoothing
        # y_train = label_smooth(y_train, smoothing=0.05, num_classes=num_classes)

        # create TensorDataset and DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # inner LOSO split for testing & validation data
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

            model = AccModel(in_channels, num_classes)
            optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=l2_lambda)
            print(f"Number of trainable parameters: {count_parameters(model)}")

            # early stopping params
            best_val_loss = float('inf')
            counter = 0

            # training loop
            for epoch in range(n_epochs):

                # loss_epoch = 0.0
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

                    # loss_epoch += loss.item()

                    print(f'Test session: S{s + 1}, Batch: [{batch_idx + 1}/{len(train_loader)}], '
                          f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {loss.item():.4f}')

                # # track losses for plotting
                # loss_epoch /= len(train_loader)
                # train_losses.append(loss_epoch)

                # validation on whole validation set after each epoch
                model.eval()
                
                with torch.no_grad():
                    pred_val = model(X_val)
                    val_loss = loss_func(pred_val, y_val)
                    
                    print(f'Test session: S{s + 1}, Epoch [{epoch + 1}/{n_epochs}], Validation Loss: {val_loss.item():.4f}')

                # val_losses.append(val_loss.item())
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    torch.save(model.state_dict(), f'../models/activity_moderntcn_S{s + 1}.pth')

                else:
                    counter += 1
                    if counter >= patience:
                        print("EARLY STOPPING - onto next split")
                        break

            # plt.plot(train_losses, label='Train Loss', color='blue')
            # plt.plot(val_losses, label='Validation Loss', color='black')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.legend()
            # plt.show()

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
    dict = load_dict(filename='ppg_dalia_dict_ppg_crm_v2')

    # ignore transient periods (not assigned an activity) to assign labels
    act_dict, in_channels = extract_activity(dict, sessions)

    # normalise
    act_dict = z_normalise(act_dict, sessions)

    # train model
    train_model(act_dict, sessions, in_channels)




if __name__ == '__main__':
    main()


