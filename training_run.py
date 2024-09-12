'''
Main script for training temporal attention model
'''

import numpy as np
import pickle
from sklearn.utils import shuffle
import time
from temporal_attention_model import TemporalConvolution, TemporalAttentionModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



def temporal_pairs(dict, sessions):
    '''
    Create temporal pairs between adjacent windows for all data
    :param dict: dictionary of all session data - each session shape (n_windows, n_channels, n_samples)
    :param sessions: list of session names
    :return x_all: temporal pairs of each session as a list of length n_sessions - each entry shape (n_windows, 1, 256, 2)
    :return y_all: ground truth HR labels as a list of length n_sessions
    :return act_all: activity labels as a list of length n_sessions
    '''

    x_all = []
    y_all = []
    act_all = []

    for s in sessions:

        x = dict[s]['bvp']

        # pair adjacent windows (i, i-1)
        x_pairs = (np.expand_dims(x[1:,:],axis=-1) , np.expand_dims(x[:-1,:],axis=-1))
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
    batch_size = 256            # number of windows to be processed together
    n_splits = 4

    # create model instance
    model = TemporalAttentionModel()
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

        # convert to torch tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # create TensorDataset and DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # create inner LOSO split to get testing & validation split
        for s in split:

            # set current session to test data
            X_test = x[s]
            y_test = y[s]
            act_test = act[s]

            # set validation data (remainder of current split)
            val_idxs = np.array([j for j in split if j != s])
            X_val = np.concatenate([x[j] for j in val_idxs], axis=0)
            y_val = np.concatenate([y[j] for j in val_idxs], axis=0)
            act_val = np.concatenate([act[j] for j in val_idxs], axis=0)

            # training loop
            for epoch in range(n_epochs):

                # create batches of windows to pass through model
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    ## batch norm??

                    # model input shape is (batch_size, n_channels, sequence_length) = (256, 1, 256)
                    x_cur = X_batch[:,:,:,0]
                    x_prev = X_batch[:,:,:,-1]

                    # forward pass x_bvp_i (x_cur) and x_bvp_i-1 (x_prev) through convolutions and then attention block
                    x_cur, x_prev = model(x_cur, x_prev)


                    # # compute loss
                    # loss = model.loss_func(X_est, y)
                    # # backprop
                    # optimizer.zero_grad()
                    # loss.backward()
                    # optimizer.step()

                    print(f'Session S{s+1}, Batch: [{1}],'
                          f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')



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

