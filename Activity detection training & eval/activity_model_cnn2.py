'''
CNN model architecture, v2
PPG-NeXt architecture (based on Google's Inception)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleBlock(nn.Module):
    '''
    Single Inception-style block architecture
    '''
    def __init__(self, in_channels, n_filters, kernel_size=[3,5,7]):

        super().__init__()

        # parallel convolutions: each branch 1x1 followed by 1xn
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_channels,out_channels=n_filters,kernel_size=1),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(in_channels=n_filters, out_channels=n_filters,
                          kernel_size=ks, padding=(ks - 1) // 2 if ks % 2 == 0 else ks // 2))
            for ks in kernel_size
        ])

        self.conv1x1 = nn.Conv1d(in_channels=n_filters, out_channels=in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, X):

        print(X.shape)
        # pass in parallel through each branch
        out = [branch(X) for branch in self.branches]
        # sum up the result
        out = self.bn1(torch.sum(torch.stack(out), dim=0))
        # upscale to match input
        out = self.conv1x1(out)
        # add residual
        out = F.relu(self.bn2(out + X))
        print(out.shape)

        return X


class MultiKernel(nn.Module):
    '''
    Series of repeating Inception-style blocks
    '''

    def __init__(self):

        super().__init__()

        n_filters = 4
        n_out = 256
        n_blocks = 3

        # create stacked structure of Inception blocks
        self.blocks = nn.ModuleList([
            SingleBlock(in_channels=n_out, n_filters=n_filters)
            for _ in range(n_blocks)
        ])

        self.bn = nn.BatchNorm1d(n_out)


    def forward(self, X):

        # iterate over each block in the ModuleList
        for i, block in enumerate(self.blocks):
            print(i)
            X = block(X)


        return X

class ConvBlock(nn.Module):
    '''
    Input convolutional block
    '''
    def __init__(self):

        super().__init__()

        in_channels = 3
        pooling_size = 3
        n_out = 256

        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=n_out,
                                     kernel_size=1,stride=1)

        self.bn = nn.BatchNorm1d(num_features=n_out)
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
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)  # global average pooling

    def forward(self, X):

        # pass through initial convolutional block
        X = self.conv(X)
        # multi-kernel blocks
        X = self.multi_kernel(X)
        # global average pooling
        X = torch.squeeze(self.gap(X), dim=-1)
        # FCN to output
        X = self.fc(X)

        print(X.shape)

        return X





