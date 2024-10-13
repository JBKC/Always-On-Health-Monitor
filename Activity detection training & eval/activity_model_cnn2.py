'''
CNN model architecture, v2
PPG-NeXt architecture
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow.raft import ResidualBlock


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
                      kernel_size=(1,1) if i == 0 else (1,3),
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

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, n_activities)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, X):

        X = F.relu(self.fc1(X))
        X = self.fc2(self.dropout(X))

        return X


class ResidualBlock(nn.Module):

    def __init__(self, cardinality=3, kernel_sizes=[3,5,7], stride=1):

        super().__init__()

        assert len(kernel_sizes) == cardinality, "Cardinality should match the number of kernel sizes provided"

        # create repeating branch of 2 convolution ResNeXt-style connections: 1x1 followed by 1xn_i where n = kernel_sizes
        # conv -> bn -> relu
        self.conv_branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                    kernel_size=1, stride=stride, padding=ks // 2),
                          nn.BatchNorm1d(out_channels),
                          nn.ReLU(),
                          nn.Conv1d(in_channels, out_channels,
                                    kernel_size=ks, stride=stride, padding=ks // 2),
                          nn.BatchNorm1d(out_channels),
                          )
            for ks in kernel_sizes
        ])

        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=3,
                                     kernel_size=1, stride=1, padding='same')
    def forward(self, X):

        # apply each branching convolution in parallel & concatenate
        out = sum([branch(X) for branch in self.conv_branches])

        out = torch.stack([branch(X) for branch in self.conv_branches], dim=0)
        out = self.conv1(out)
        out = self.conv1(X) + out

        return out

class AccModel(nn.Module):
    '''
    Model architecture that takes only accelerometer channels
    input shape = (batch_size, n_channels, n_samples)
    '''
    def __init__(self,in_channels=3, n_filters=8, pool_size=2):

        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=n_filters,
                                     kernel_size=1, stride=1, padding='same')

        self.bn = nn.BatchNorm1d(n_filters)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        self.res_block = ResidualBlock()

        self.linear = FCN()

    def forward(self, X):

        print(X.shape)
        # implement first block
        X = self.bn(self.conv1(X))
        X = self.pool(F.relu(X))
        print(X.shape)

        # ResNext-style block x3 sequential
        for i in range(3):
            X = self.res_block(X)

        return X






