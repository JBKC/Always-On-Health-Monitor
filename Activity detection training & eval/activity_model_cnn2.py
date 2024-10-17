'''
CNN model architecture, v2
PPG-NeXt architecture (based on Google's Inception)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F





class MultiKernel(nn.Module):
    '''
    Series of repeating Inception-style blocks
    '''

    def __init__(self):

        super().__init__()

        in_channels = 3
        n_filters = 32
        n_blocks = 3
        pooling_size = 3
        n_out = 128

        # create stacked structure of Inception blocks
        self.conv_blocks = nn.ModuleList([
            Inception(in_channels=in_channels if i == 0 else n_out, n_filters=n_filters,
                      pooling_size=pooling_size, stride=stride)
            for i in range(n_blocks)
        ])

        self.res1x1 = nn.Conv1d(in_channels=in_channels,out_channels=n_out,
                                     kernel_size=1,stride=stride)

        self.bn = nn.BatchNorm1d(n_out)


    def forward(self, X):

        # iterate over each block in the ModuleList
        for i, block in enumerate(self.conv_blocks):
            X = block(X)


        return X

class ConvBlock(nn.Module):
    '''
    Input convolutional block
    '''
    def __init__(self):

        super().__init__()

        in_channels = 3
        n_filters = 32
        pooling_size = 3

        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=n_filters,
                                     kernel_size=1,stride=1)

        self.bn = nn.BatchNorm1d(num_features=n_filters)
        self.pooling = nn.MaxPool1d(kernel_size=pooling_size,
                                    padding=(pooling_size-1)//2 if pooling_size % 2 == 0 else pooling_size//2)

    def forward(self, X):

        print(X.shape)
        X = self.bn(self.conv1(X))
        X = self.pooling(F.relu(X))
        print(X.shape)

        return X


class AccModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples)
    '''
    def __init__(self):
        super().__init__()

        n_activities = 8

        self.conv = ConvBlock()
        self.multi_kernel = MultiKernel()

    def forward(self, X):

        # pass through initial convolutional block
        X = self.conv(X)

        # multi-kernel blocks
        X = self.multi_kernel(X)

        return X





