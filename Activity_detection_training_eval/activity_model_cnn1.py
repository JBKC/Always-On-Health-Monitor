'''
CNN model architecture, v1
Based on Inception v1 / v2
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleBlock(nn.Module):
    '''
    Single Inception-style block architecture
    '''

    def __init__(self, n_filters, kernel_size=[3, 5, 7], pooling_size=2):
        super().__init__()

        # parallel convolutions: each branch 1x1 followed by 1xn
        self.branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=1, stride=1),
                nn.BatchNorm1d(n_filters),
                nn.ELU(),
                nn.Conv1d(in_channels=n_filters, out_channels=n_filters,
                          kernel_size=ks, stride=2, padding=(ks - 1) // 2 if ks % 2 == 0 else ks // 2))
            for ks in kernel_size
        ])

        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size,padding=0)
        self.conv1x1 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=1)
        self.bn = nn.BatchNorm1d(n_filters*4)

    def forward(self, X):

        # pass in parallel through each branch
        out = [b(X) for b in self.branch]
        print(f'Branches shape: {[i.shape for i in out]}')
        out = torch.cat(out, dim=1)
        print(f'All branches shape: {out.shape}')

        # pool branch
        pool_out = self.pooling(X)
        print(f'Pool shape: {pool_out.shape}')

        # depth concatenate
        out = F.elu(self.bn(torch.cat((out, pool_out), dim=1)))

        print(f'Multi_kernel out shape: {out.shape}')

        return out


class MultiKernel(nn.Module):
    '''
    Series of repeating Inception-style blocks
    '''

    def __init__(self, n_filters, n_blocks):
        super().__init__()

        # create stacked structure of Inception blocks
        self.blocks = nn.ModuleList([
            SingleBlock(n_filters=n_filters*4**(i))
            for i in range(n_blocks)
        ])

    def forward(self, X):
        # iterate over each block in the ModuleList
        for i, block in enumerate(self.blocks):
            X = block(X)

        return X


class InitialBlock(nn.Module):
    '''
    Input convolutional block, pre-Inception blocks
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        pooling_size = 3

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=2,
                               padding=(kernel_size - 1) // 2 if kernel_size % 2 == 0 else kernel_size // 2)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=2, padding=1)

    def forward(self, X):
        print(f'Input shape: {X.shape}')
        X = self.bn(self.conv1(X))
        X = self.pooling(F.elu(X))
        print(f'Initial_block out shape: {X.shape}')

        return X


class AccModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples)
    '''

    def __init__(self, in_channels, num_classes):
        super().__init__()

        n_filters = 32                              # number of initial filters
        n_blocks = 3                                # number of repeating Inception blocks
        gap_out = n_filters*4**(n_blocks)           # number of channels in GAP output

        self.initial_block = InitialBlock(in_channels=in_channels, out_channels=n_filters)
        self.multi_kernel = MultiKernel(n_filters=n_filters, n_blocks=n_blocks)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)  # global average pooling
        self.fc = nn.Linear(gap_out, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, X):
        # pass through initial convolutional block
        X = self.initial_block(X)
        # multi-kernel blocks
        X = self.multi_kernel(X)
        # global average pooling
        X = torch.squeeze(self.gap(X), dim=-1)
        print(f'Post-GAP shape: {X.shape}')
        # dropout
        X = self.dropout(X)
        # FCN to output
        X = self.fc(X)
        print(f'Model output shape: {X.shape}')

        return X





