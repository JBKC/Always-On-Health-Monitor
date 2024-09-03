'''
Adaptive linear model
Takes in accelerometer data as input into a CNN
'''

import numpy as np
import requests
import torch
import torch.nn as nn

class AdaptiveLinearModel(nn.Module):
    def __init__(self, n_epochs=500, input_shape=(3,256,1)):
        super().__init__()

        self.n_epochs = n_epochs

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, kernel_size=(3, 21), padding='same')

        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=1, kernel_size=(3, 1), padding='valid')
    def forward(self,x):
        # pass through model layers

        # pass through 1st conv layer
        x = self.conv1(x)
        # pass through 2nd conv layer
        x = self.conv2(x)

        # returns data of shape (1,256) - since dimensions 1 (n_channels) and 3 (n_windows) are now both = 1
        return x[:, 0, :, 0]

    def adaptive_loss(self, y_true, y_pred):
        # define custom loss function: MSE( FFT(CNN output) , FFT(raw PPG input) ) = MSE ( FFT(y_pred), FFT(y_true) )

        y_true = y_true[:, 0, :, 0]             # match output of conv layers

        # take FFT
        y_true_fft = torch.fft.fft(y_true)
        y_pred_fft = torch.fft.fft(y_pred)

        # calculate error (raw ppg - motion artifact estimate)
        e = torch.abs(y_true_fft - y_pred_fft)
        # MSE
        e = torch.sum(e ** 2, dim=-1)

        return torch.mean(e)

    def loss(self):
        return
