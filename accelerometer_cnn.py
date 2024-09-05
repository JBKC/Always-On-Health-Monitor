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

        return X

    def reinitialize_weights(self):
        self.load_state_dict(self.initial_state)

    def train_batch(self, X, session, batch, n_epochs, optimizer):
        '''
        Perform training for one batch
        :param X: shape (batch_size, 4, 256) - combined accelerometer and PPG data
        :param optimizer: PyTorch optimizer
        :param n_epochs: number of epochs to train
        :return: filtered PPG data
        '''
        self.train()

        # accelerometer data are inputs
        x = X[:, :, 1:, :]  # (batch_size, 1, 3, 256)
        # PPG data are targets
        y_true = X[:, :, :1, :]  # (batch_size, 1, 1, 256)

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # forward pass
            y_pred = self(x)  # Forward pass
            # calculate loss
            loss = self.adaptive_loss(y_true, y_pred)
            # backpass
            loss.backward()
            optimizer.step()

            # subtract the motion artifact estimate to extract cleaned BVP
            x_out = y_true[:, 0, 0, :] - y_pred

            if epoch % 10 == 0:
                print(f'Session {session}, Batch: {batch+1}, '
                      f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')


        self.eval()

        x_out = y_true[:, 0, 0, :] - y_pred

        # get signal into original shape: (n_windows, 1, 256)
        x_out = x_out[:, 0, :, :]
        print(x_out.shape)

        # Reset the weights
        self.reinitialize_weights()

        return x_out


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


