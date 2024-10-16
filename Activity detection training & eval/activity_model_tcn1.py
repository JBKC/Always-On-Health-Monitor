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

    def __init__(self, in_channels, n_filters, pooling_size, stride, kernel_size=[10,20,40]):

        super().__init__()

        # 1x1 bottleneck
        self.conv1x1 = nn.Conv1d(in_channels=in_channels,out_channels=n_filters,
                                     kernel_size=1,stride=stride)
        # parallel convolutions
        self.branches = nn.ModuleList([nn.Conv1d(
            in_channels=n_filters,out_channels=n_filters,
            kernel_size=ks,stride=stride,padding=(ks-1)//2 if ks % 2 == 0 else ks//2) for ks in kernel_size])

        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=stride,
                                    padding=(pooling_size-1)//2 if pooling_size % 2 == 0 else pooling_size//2)

    def forward(self, X):

        print(X.shape)
        # pass through bottleneck
        X1x1 = self.conv1x1(X)
        print(X1x1.shape)

        # pass through convolution branches in parallel to get list
        out = [branch(X1x1) for branch in self.branches]

        for i in range(len(out)):
            print(out[i].shape)

        # pass through maxpool branch
        max_out = self.conv1x1(self.pooling(X))
        print(max_out.shape)
        out.append(max_out)

        # concatenate outputs across channel dimension
        X = torch.cat(out, dim=1)
        print(X.shape)


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
        pooling_size = 3

        # create stacked structure of Inception blocks
        self.conv_blocks = nn.ModuleList([
            Inception(in_channels=in_channels if i == 0 else n_filters, n_filters=n_filters,
                       pooling_size=pooling_size, stride=stride)
            for i in range(n_blocks)
        ])

    def forward(self, X):

        # iterate over each block in the ModuleList
        for block in self.conv_blocks:
            X = block(X)


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






