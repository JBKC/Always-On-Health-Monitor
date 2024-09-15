'''
Generate dataset with artificially sped up samples to help model predict high HR cases
Combine this dataset with the original & noise datasets to form the final dataset
'''

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.utils import shuffle


class GenerateFullDataset(Dataset):
    def __init__(self, X, y, X_noise, y_noise, ratio_sampling=0.5):
        '''
        __init__ is run when instance of class is created
        :param X:
        :param y:
        :param X_noise:
        :param y_noise:
        :param ratio_sampling:
        '''

        # convert input data into torch
        self.X_in = torch.from_numpy(X).float()
        self.y_in = torch.from_numpy(y).float()

        self.X_noise_in = torch.from_numpy(X_noise).float()
        self.y_noise_in = torch.from_numpy(y_noise).float()

        self.ratio_sampling = ratio_sampling        # % of random samples taken from adversarial dataset
        self.fs = 32

        self.create_sped()
        self.setup_clean_indexes()
        self.combine_datasets()
        self.shuffle_dataset()

    def __getitem__(self, index):
        # returns a single window & corresponding label from dataset at given index
        return self.X_out[index], self.y_out[index]

    def __len__(self):
        # returns total number of windows
        return self.X_out.shape[0]

    def create_sped(self):
        
        # create artificially sped up signal, by offsetting (for added variation) -> combining -> downsampling
        self.X_sped = torch.cat([self.X_in[:-4, ...], self.X_in[4:, ...]], dim=1)[:, ::2, :]
        # align labels
        self.y_sped = 2 * self.y_in[:-4]

        # only keep windows that have sped up HR < 300bpm
        mask = self.y_sped.flatten() < 300
        self.X_sped = self.X_sped[mask]
        self.y_sped = self.y_sped[mask]

    def setup_clean_indexes(self):
        # find "clean" PPG windows - ie ones where the dominant frequency closely matches the ground truth label

        # get FFT
        fft_sped = torch.abs(torch.fft.fft(self.X_sped[:, 0, :]))[:,:128].T
        # get dominant frequency & convert to Hz
        freq_dom = torch.argmax(fft_sped, dim=0) * 7.5

        # get indices of windows with "clean" signals
        self.clean_indexes = torch.where(torch.abs(freq_dom.flatten() - self.y_sped.flatten()) < 10)[0]

    def combine_datasets(self):
        # combine all 3 datasets to create the final dataset

        # calculate number of random samples to take
        n_random_samples = int(self.ratio_sampling * self.y_in.size(0))
        # get indices for adversarial dataset
        indexes = torch.randperm(self.y_noise_in.size(0))[:n_random_samples]

        # extract random samples from high HR dataset
        X_highhr = self.X_sped[self.clean_indexes.flatten()]
        y_highhr = self.y_sped[self.clean_indexes.flatten()]

        # concatenate all datasets together
        self.X_out = torch.cat([self.X_in, self.X_noise_in[indexes], X_highhr], dim=0)
        self.y_out = torch.cat([self.y_in, self.y_noise_in[indexes], y_highhr], dim=0)


    def shuffle_dataset(self):
        # shuffle dataset randomly

        perm = torch.randperm(self.X_out.size(0))
        self.X_out = self.X_out[perm]
        self.y_out = self.y_out[perm]

    def on_epoch_end(self):
        # regenerate the random y_noise labels after each epoch

        self.y_noise_in = torch.from_numpy(np.random.uniform(low=20, high=300, size=(self.X_noise_in.shape[0], 1))).float()
        self.combine_datasets()
        self.shuffle_dataset()