'''
CNN model architecture, v1
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ConvBlock(nn.Module):
    '''
    Repeating convolution layer structure
    '''

    def __init__(self, in_channels, n_filters, n_layers=5, pool_size=(1,2), kernel_size=(1,3)):

        super().__init__()

        # conv block = conv + BN + ReLU + pooling
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    stride=(1,1),
                    padding='same'
                ),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size)
            ) for i in range(n_layers)]
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):


        return


class ConvLayers(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=1, n_filters=32, pool_size=4)
        self.conv_block2 = ConvBlock(in_channels=32, n_filters=48, pool_size=2)
        self.conv_block3 = ConvBlock(in_channels=48, n_filters=64, pool_size=2)



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


        self.bn1 = nn.BatchNorm2d(16)


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




