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
        self.prediction_history = []

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(3, 21), padding='same')
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1,
                               kernel_size=(3, 1), padding='valid')

    def forward(self, X):
        '''
        Define forward pass of adaptive filter model
        :param X: shape (n_windows,3,256)
        :return:
        '''

        self.train()

        X = self.conv1(X)               # 1st conv layer
        # no specified activation function (linear)
        X = self.conv2(X)               # 2nd conv layer

        return X


    def adaptive_loss(self, y_true, y_pred):
        # define custom loss function:
        # MSE( FFT(CNN output) , FFT(raw PPG input) ) == MSE ( FFT(y_pred), FFT(y_true) )

        # remove redundant dimensions
        y_true = y_true[:, 0, 0, :]
        y_pred = y_pred[:, 0, 0, :]

        # take FFT
        y_true_fft = torch.fft.fft(y_true)
        y_pred_fft = torch.fft.fft(y_pred)

        # calculate error (raw ppg - motion artifact estimate)
        e = torch.abs(y_true_fft - y_pred_fft)
        # MSE
        e = torch.sum(e ** 2, dim=-1)

        return torch.mean(e)


