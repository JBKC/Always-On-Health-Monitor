'''
Adaptive linear model
Takes in accelerometer data as input into a CNN
Input shape = (3,256,1)
'''

import numpy as np
import requests
import torch
import torch.nn as nn

class AdaptiveLinearModel(nn.Module):
    def __init__(self, n_epochs=500):
        super().__init__()

        self.n_epochs = n_epochs
        self.prediction_history = []

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, kernel_size=(3, 21), padding='same')
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=1, kernel_size=(3, 1), padding='valid')

    def forward(self,x):
        # define forward pass steps

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

    def grad(self, inputs, targets):
        # compute gradients

        self.local_optimizer.zero_grad()                        # reset all gradients to zero with each backpass
        loss_value = self.adaptive_loss(inputs, targets)        # calculate loss
        loss_value.backward()                                   # backprop
        return loss_value

    def call(self, inputs):
        # define training loop

        x = inputs[:, 1:, ...]
        y = inputs[:, :1, ...]

        self.train()                # set model to training mode

        # run training loop for specified number of epochs
        for epoch in range(self.n_epochs):

            self.grad(x, y)
            self.local_optimizer.step()             # update model parameters

            # keep track of predictions
            if self.track_prediction_history:
                x_out = y[:, 0, :, 0] - self(x)
                self.prediction_history.append(x_out.detach().cpu().numpy())

        # evaluate final prediction
        x_out = y[:, 0, :, 0] - self(x)
        self.eval()

        return x_out
