'''
Adaptive linear filter
Takes in accelerometer data as input into a CNN, where the adaptive linear filter is the loss function
Input shape = (3,256,1)
'''

import numpy as np
import torch
import torch.nn as nn

class AdaptiveLinearModel(nn.Module):
    def __init__(self, n_epochs):
        super().__init__()

        self.n_epochs = n_epochs

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(3, 21), padding='same')
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(3, 1), padding='valid')

        self.initial_state = self.state_dict()          # initial weights of model

    def forward(self, X):
        '''
        Define forward pass of adaptive filter model
        :param X: shape (batch_size, n_windows,4,256)
        :return:
        '''

        self.train()

        X = self.conv1(X)                   # 1st conv layer
        # no specified activation function (linear)
        X = self.conv2(X)              # 2nd conv layer

        # remove redundant dimensions
        return X[:, 0, 0, :]

    def adaptive_loss(self, y_true, y_pred):
        '''
        Apply adaptive filter
        custom loss function: MSE( FFT(CNN output) , FFT(raw PPG signal) )
        :param y_true: raw PPG signal
        :param y_pred: predicted motion artifacts from CNN
        :return: mean squared error between y_true and y_pred
        '''

        # remove redundant dimensions
        y_true = y_true[:, 0, 0, :]

        # take FFT
        y_true_fft = torch.fft.fft(y_true)
        y_pred_fft = torch.fft.fft(y_pred)

        # calculate error (raw ppg - motion artifact estimate)
        e = torch.abs(y_true_fft - y_pred_fft)
        # single MSE for each batch
        e = torch.sum(e ** 2, dim=-1)

        # single MSE value across all batches
        return torch.mean(e)


