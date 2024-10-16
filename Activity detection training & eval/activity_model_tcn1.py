'''
TCN model architecture, v1
Based on InceptionTime
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    '''
    Single Inception block
    '''

    def __init__(self, in_channels, n_filters, pooling, stride, kernel_size=[10,20,40]):

        super().__init__()

        # 1x1 bottleneck
        self.conv1x1 = nn.Conv1d(in_channels=in_channels,out_channels=n_filters,
                                     kernel_size=1,stride=stride,padding='same')
        # parallel convolutions
        self.convincept = nn.Conv1d(in_channels=n_filters,out_channels=n_filters, )

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


class ConvBlocks(nn.Module):
    '''
    Series of repeating Inception blocks
    '''

    def __init__(self):

        super().__init__()

        in_channels = 3
        n_filters = 32
        n_blocks = 6
        stride = 1
        pooling = 3

        # create stacked structure of Inception blocks
        self.conv_blocks = nn.ModuleList([
            Inception(in_channels=in_channels if i == 0 else n_filters, n_filters=n_filters,
                       pooling=pooling, stride=stride)
            for i in range(n_blocks)
        ])

    def forward(self, X):

        X = self.conv_blocks(X)


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


class TCNModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples)
    '''
    def __init__(self):
        super().__init__()

        self.conv = ConvBlocks()

    def forward(self, X):

        # pass through inception time blocks
        X = self.conv(X)

        return X






