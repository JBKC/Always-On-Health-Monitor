'''
Main script for training temporal attention model
'''

import numpy as np
import pickle
from sklearn.utils import shuffle
import time
from temporal_attention_model import TemporalAttentionModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal


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

    def NLL(dist, y):
        '''
        Negative log likelihood loss of observation y, given distribution dist
        :param dist: predicted Gaussian distribution
        :param y: ground truth label
        :return: NLL for each window
        '''

        return -dist.log_prob(y)

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

    # outer LOSO split for training data
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

        # inner LOSO split for testing & validation data
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

            # convert to torch tensors
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

            # training loop
            for epoch in range(n_epochs):

                model.train()

                # create training batches of windows to pass through model
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

                    optimizer.zero_grad()

                    # prep data for model input - shape is (batch_size, n_channels, sequence_length) = (256, 1, 256)
                    x_cur = X_batch[:,:,:,0]
                    x_prev = X_batch[:,:,:,-1]

                    # forward pass through model (convolutions + attention + probabilistic)
                    dist = model(x_cur, x_prev)

                    # calculate training loss on distribution
                    loss = NLL(dist, y_batch).mean()
                    loss.backward()
                    optimizer.step()

                    print(f'Test session: S{s+1}, Batch: [{batch_idx + 1}/{len(train_loader)}], '
                          f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {loss.item():.4f}')


                # validation on whole validation set after each epoch
                model.eval()

                with torch.no_grad():
                    val_dist = model(X_val[:,:,:,0], X_val[:,:,:,-1])
                    val_loss = NLL(val_dist, y_val).mean()          # average validation across all windows

                    print(f'Test session: S{s+1}, Epoch [{epoch + 1}/{n_epochs}], Validation Loss: {val_loss.item():.4f}')

            # test on held-out session
            with torch.no_grad():
                test_dist = model(X_test[:,:,:,0], X_test[:,:,:,-1])
                test_loss = NLL(test_dist, y_test).mean()
                print(f'Test session: S{s+1}, Test Loss: {test_loss.item():.4f}')

    end_time = time.time()
    print("TRAINING COMPLETE: time ", (end_time - start_time) / 3600, " hours.")



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

