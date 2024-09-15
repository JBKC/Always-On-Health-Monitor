'''
Main script for training temporal attention model
Full training dataset combines original x_BVP PPG Dalia dataset with adversarial examples & high HR examples
'''

import numpy as np
import pickle
from sklearn.utils import shuffle
import time
from temporal_attention_model import TemporalAttentionModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from generate_high_HR_dataset import GenerateFullDataset

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

def train_model(dict, noise_dict, sessions):
    '''
    Create Leave One Session Out split and run through model
    :param dict: dictionary of all session data - each session shape (n_windows, n_channels, n_samples)
    :param dict: dictionary of adversarial noise data - each session shape (n_windows, n_channels, n_samples)
    :param sessions: list of session names
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
    patience = 10               # early stopping parameter
    batch_size = 256            # number of windows to be processed together
    n_splits = 4

    # create model instance
    model = TemporalAttentionModel()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)

    # create temporal pairs of time windows for both original data and noise data
    x, y, act = temporal_pairs(dict, sessions)
    x_noise, y_noise, _ = temporal_pairs(noise_dict, sessions)

    # LOSO splits
    ids = shuffle(list(range(len(sessions))))       # index each session
    splits = np.array_split(ids, n_splits)

    # Load checkpoint if available
    checkpoint_path = '/models/best_temporal_attention_model.pth'

    try:
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        counter = checkpoint['counter']
        splits = checkpoint['splits'],
        processed_splits = checkpoint['processed_splits']
        last_split_idx = checkpoint['last_split_idx']
        last_session = checkpoint['last_session']
        last_session_idx = checkpoint['last_session_idx']
        print(f"Checkpoint found, resuming from Split {last_split_idx + 1}, Session {last_session + 1}")

    except FileNotFoundError:
        print("No checkpoint found, training from scratch")
        best_val_loss = float('inf')        # early stopping parameter
        counter = 0                         # early stopping parameter
        processed_splits = []               # track each split as they are processed
        last_split_idx = -1
        last_session_idx = -1
        epoch = -1

    start_time = time.time()

    # outer LOSO split for training data
    for split_idx, split in enumerate(splits):

        # skip already-processed splits
        if split_idx <= last_split_idx:
            continue

        # set training data (current split = testing/validation data)
        train_idxs = np.array([i for i in ids if i not in split])

        X_train = np.concatenate([x[i] for i in train_idxs], axis=0)
        y_train = np.concatenate([y[i] for i in train_idxs], axis=0)
        act_train = np.concatenate([act[i] for i in train_idxs], axis=0)

        X_noise_train = np.concatenate([x_noise[i] for i in train_idxs], axis=0)
        y_noise_train = np.concatenate([y_noise[i] for i in train_idxs], axis=0)

        # inner LOSO split for testing & validation data
        for session_idx, s in enumerate(split):

            # skip already-processed sessions in current split
            if split_idx == last_split_idx and session_idx <= last_session_idx:
                continue

            # set current session's original data to test data
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
            for epoch in range(epoch +1, n_epochs):
                model.train()

                # combine datasets for training
                X_train, y_train = GenerateFullDataset(X_train, y_train, X_noise_train, y_noise_train, batch_size=batch_size)

                # create TensorDataset and DataLoader for batching
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

                    print(f'Test session: S{s + 1}, Batch: [{batch_idx + 1}/{len(train_loader)}], '
                          f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {loss.item():.4f}')


                # validation on whole validation set after each epoch
                model.eval()

                with torch.no_grad():
                    val_dist = model(X_val[:,:,:,0], X_val[:,:,:,-1])
                    val_loss = NLL(val_dist, y_val).mean()          # average validation across all windows

                    print(f'Test session: S{s + 1}, Epoch [{epoch + 1}/{n_epochs}], Validation Loss: {val_loss.item():.4f}')

                # early stopping criteria
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0

                    # save down checkpoint of current best model state
                    checkpoint = {
                        'model_state_dict': model.state_dict(),  # model weights
                        'optimizer_state_dict': optimizer.state_dict(),  # optimizer state
                        'epoch': epoch,  # save the current epoch
                        'best_val_loss': best_val_loss,  # the best validation loss
                        'counter': counter,  # early stopping counter
                        'splits': splits,  # training splits
                        'processed_splits': processed_splits,  # track which splits have already been processed
                        'last_split_idx': split_idx,  # the index of the last split
                        'last_session': s,  # the last session in the current split
                        'last_session_idx': session_idx,  # the index of the last session in the current split
                    }
                    torch.save(checkpoint, checkpoint_path)

                else:
                    counter += 1
                    if counter >= patience:
                        print("EARLY STOPPING - onto next split")
                        break

            # test on held-out session after all epochs complete
            with torch.no_grad():
                test_dist = model(X_test[:,:,:,0], X_test[:,:,:,-1])
                test_loss = NLL(test_dist, y_test).mean()
                print(f'Test session: S{s + 1}, Test Loss: {test_loss.item():.4f}')

        # mark current split as processed
        processed_splits.append(split_idx)
        print(f"Split {split_idx + 1} processed.")

    end_time = time.time()
    print("TRAINING COMPLETE: time ", (end_time - start_time) / 3600, " hours.")

    return



def main():

    def load_dict(filename):

        with open(filename, 'rb') as file:
            data_dict = pickle.load(file)
            print(f'Data dictionary loaded from {filename}')

            return data_dict

    sessions = [f'S{i}' for i in range(1, 16)]

    # load original & adversarial data
    dict = load_dict(filename='ppg_filt_dict')
    noise_dict = load_dict(filename='noise_dict')

    train_model(dict, noise_dict, sessions)


if __name__ == '__main__':
    main()

