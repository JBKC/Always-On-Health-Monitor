'''
Generate dataset with artificially sped up samples to help model predict high HR cases
Combine this dataset with the noise dataset (from generate_adversarial_dataset) and original data to form the final dataset
Uses torch.utils.data Dataset to automatically create batches
'''

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.utils import shuffle


class GenerateFullDataset(Dataset):
    def __init__(self, X, y, X_noise, y_noise, ratio_sampling=0.1):
        '''
        __init__ is run when instance of class is created
        :param X: training data across all sessions in training split, shape (n_windows, 256, 2)
            where the final dimension is (x_cur, x_prev)
        :param y: training labels, shape (n_windows,)
        :param X_noise: noise data, shape (n_windows, 256, 2)
        :param y_noise: noise labels data, shape (n_windows,)
        :param ratio_sampling: % of total noise data windows to randomly sample
        '''

        # convert input data into torch
        self.X_in = torch.from_numpy(X).float()
        self.y_in = torch.from_numpy(y).float()

        self.X_noise_in = torch.from_numpy(X_noise).float()
        self.y_noise_in = torch.from_numpy(y_noise).float()

        self.n_random_samples = int(ratio_sampling * self.y_in.shape[0])
        self.freq_bin = 32 / 256                    # fs / n_samples per window

        # execute methods
        self.create_sped()
        self.find_clean_idxs()
        self.combine_datasets()
        self.shuffle_dataset()

    def __getitem__(self, index):
        # fetches values of X training data and y labels at a single index
        return self.X_out[index], self.y_out[index]

    def __len__(self):
        # returns total number of windows in dataset
        return self.X_out.shape[0]

    def create_sped(self):
        '''
        Artificially speed up every signal in X_in
        done by concatenating 2 parts of the signal together, then downsampling by a factor of 2
        '''

        offset = 4              # window offset
        # combine current window with an offset window
        self.X_sped = torch.cat([self.X_in[offset:,:,:], self.X_in[:-offset,:,:]], dim=1)
        # downsample to get double effective frequency
        self.X_sped = self.X_sped[:,::2,:]

        # align labels
        self.y_sped = 2 * self.y_in[:-offset]

        # only keep windows that have sped up HR < 300bpm
        mask = self.y_sped.flatten() < 250
        self.X_sped = self.X_sped[mask]
        self.y_sped = self.y_sped[mask]

    def find_clean_idxs(self):
        '''
        find which sped up windows can be considered "clean"
        ie. ones where the dominant frequency closely matches the ground truth label
        this is the final list of high HR examples
        '''

        tol = 10                    # tolerance for considering a sped up signal as "clean"

        # get FFT of x_cur - gives shape (128, n_windows)
        fft_sped = torch.abs(torch.fft.fft(self.X_sped[:, :, 0]))[:,:int(self.X_sped.shape[1]/2)].T
        # get dominant frequency bin & convert to BPM
        freq_dom = torch.argmax(fft_sped, dim=0) * self.freq_bin * 60

        # get indices of windows with "clean" signals
        self.clean_idxs = torch.where(torch.abs(freq_dom - self.y_sped) < tol)[0]

    def combine_datasets(self):
        '''
        Combine original X, X_noise and X_highhr to create final dataset
        '''

        # get indices to pull random data from adversarial dataset
        idxs = np.arange(self.y_noise_in.shape[0])
        idxs = np.random.choice(idxs, size=self.n_random_samples)

        # pull clean high HR examples
        X_highhr = self.X_sped[self.clean_idxs.flatten()]
        y_highhr = self.y_sped[self.clean_idxs.flatten()]

        print(len(self.X_in))
        print(len(self.X_noise_in[idxs]))
        print(len(X_highhr))

        # concatenate all datasets together
        self.X_out = torch.cat([self.X_in, self.X_noise_in[idxs], X_highhr], dim=0)
        self.y_out = torch.cat([self.y_in, self.y_noise_in[idxs], y_highhr], dim=0)

    def shuffle_dataset(self):
        '''
        Randomly shuffles dataset
        :return:
        '''

        perm = torch.randperm(self.X_out.shape[0])
        self.X_out = self.X_out[perm]
        self.y_out = self.y_out[perm]

        print(f'All Training Data Shape: {self.X_out.shape}')


