'''
CNN model architecture, v1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ConvBlock(nn.Module):
    '''
    Repeating convolution block structure
    '''

    def __init__(self, in_channels, n_filters, pool_size, kernel_size=5, dropout=0.5):

        super().__init__()

        self.kernel_size = kernel_size
        self.n_filters = n_filters

        # conv block = conv layer + BN + ReLu + pooling
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                ),
                nn.ReLU()
            ) for i in range(3)]
        )

        self.pool = nn.AvgPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):


        return



class ConvLayers(nn.Module):
    '''
    Series of convolution layers
    '''

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(1, 1),
                               stride=(1,1), padding='same')

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(1, 3),
                               stride=(1,1), padding='same')

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3),
                               stride=(1,1), padding='same',dilation=2)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3),
                               stride=(1, 1), padding='same', dilation=2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 3),
                               stride=(1, 1), padding='same', dilation=2)

        self.pool = nn.MaxPool2d(kernel_size=(1,2))


    def forward(self, X):

        # fuse channels
        X = F.relu(self.conv1(X))

        X = F.relu(self.conv2(X))
        X = self.pool(X)

        X = F.relu(self.conv3(X))
        X = self.pool(X)

        X = F.relu(self.conv4(X))
        X = self.pool(X)

        X = F.relu(self.conv5(X))

        X = torch.flatten(X, start_dim=1)

        return X

class FCN(nn.Module):
    '''
    Series of fully connected layers
    '''
    def __init__(self, n_activities=8):

        super().__init__()

        self.fc1 = nn.Linear(256, n_activities)

    def forward(self, X):

        X = self.fc1(X)

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
        self.linear(X)

        return




