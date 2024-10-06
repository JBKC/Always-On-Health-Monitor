'''
TCN model architecture, v1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    '''
    Single convolutional layer
    '''

    def __init__(self, in_channels, n_filters, kernel_size, pool_size=(1,2), pooling=True):

        super().__init__()

        # conv block = conv + BN + ReLU + pooling
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=n_filters,
                                     kernel_size=kernel_size,stride=(1,1),padding='same')
        self.bn = nn.BatchNorm2d(n_filters)
        self.pooling = pooling
        self.pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, X):

        X = self.conv(X)
        X = self.bn(X)
        X = F.relu(X)

        if self.pooling:
            X = self.pool(X)

        return X


class ConvLayers(nn.Module):
    '''
    Series of convolutional layers
    '''

    def __init__(self):

        super().__init__()

        filters = [8, 16, 32, 64, 16]
        in_channels = 3

        # don't apply pooling on first and last layers

        self.conv_blocks = nn.ModuleList([
            ConvLayer(in_channels=in_channels if i == 0 else filters[i - 1], n_filters=filters[i],
                      kernel_size=(1,5) if i == 0 else (1,5),
                      pooling=(i != 0 and i != len(filters)-1))
            for i in range(len(filters))
        ])

    def forward(self, X):

        for conv_block in self.conv_blocks:
            X = conv_block(X)

        X = torch.flatten(X, start_dim=1)

        return X


class FCN(nn.Module):
    '''
    Series of fully connected layers
    '''
    def __init__(self, n_activities=8):

        super().__init__()

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_activities)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, X):

        X = F.relu(self.fc1(X))
        X = self.fc2(self.dropout(X))

        return X


class AccModel(nn.Module):
    '''
    Model architecture that takes only accelerometer channels
    input shape = (batch_size, n_channels, n_samples)
    '''
    def __init__(self):
        super().__init__()

        self.convolution = ConvLayers()
        self.linear = FCN()

    def forward(self, X):

        X = self.convolution(X)
        X = self.linear(X)

        return X






