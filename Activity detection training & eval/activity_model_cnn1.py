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

    def __init__(self, in_channels, n_filters, kernel_size=[3, 5, 7], pooling_size=2):
        super().__init__()

        # parallel convolutions: each branch 1x1 followed by 1xn
        self.branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=n_filters, kernel_size=1, stride=1),
                nn.BatchNorm1d(n_filters),
                nn.ELU(),
                nn.Conv1d(in_channels=n_filters, out_channels=n_filters,
                          kernel_size=ks, stride=2, padding=(ks - 1) // 2 if ks % 2 == 0 else ks // 2))
            for ks in kernel_size
        ])

        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size,
                                    padding=(pooling_size - 1) // 2 if pooling_size % 2 == 0 else pooling_size // 2)
        self.conv1x1 = nn.Conv1d(in_channels=n_filters, out_channels=in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, X):

        # pass in parallel through each branch
        out = [b(X) for b in self.branch]
        print(f'Branches shape: {[i.shape for i in out]}')

        # sum up the result
        out = self.bn1(torch.sum(torch.stack(out), dim=0))
        # upscale to match input
        out = self.conv1x1(out)
        print(out.shape)

        # residual branch
        res = self.conv1x1(self.pooling(X))
        print(res.shape)

        # add residual
        out = F.elu(self.bn2(out + res))
        print(out.shape)

        return out


class MultiKernel(nn.Module):
    '''
    Series of repeating Inception-style blocks
    '''

    def __init__(self, bottleneck, out_channels):
        super().__init__()

        n_blocks = 3

        # create stacked structure of Inception blocks
        self.blocks = nn.ModuleList([
            SingleBlock(in_channels=out_channels, n_filters=bottleneck)
            for _ in range(n_blocks)
        ])

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        # iterate over each block in the ModuleList
        for i, block in enumerate(self.blocks):
            X = block(X)

        return X


class InitialBlock(nn.Module):
    '''
    Input convolutional block, pre-Inception blocks
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()

        pooling_size = 3

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=2, padding='valid')

        self.bn = nn.BatchNorm1d(num_features=out_channels)
        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=2, padding=0)

    def forward(self, X):
        print(f'Input shape: {X.shape}')
        X = self.bn(self.conv1(X))
        X = self.pooling(F.elu(X))
        print(f'Initial block out shape: {X.shape}')

        return X


class AccModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples)
    '''

    def __init__(self, in_channels, num_classes):
        super().__init__()

        out_channels = 256  # number of feature maps to achieve at the output of each block

        self.initial_block = InitialBlock(in_channels=in_channels, out_channels=out_channels)
        self.multi_kernel = MultiKernel(bottleneck=out_channels//4, out_channels=out_channels)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)  # global average pooling
        self.fc = nn.Linear(out_channels, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, X):
        # pass through initial convolutional block
        X = self.initial_block(X)
        # multi-kernel blocks
        X = self.multi_kernel(X)
        # global average pooling
        X = torch.squeeze(self.gap(X), dim=-1)
        # dropout
        X = self.dropout(X)
        # FCN to output
        X = self.fc(X)

        return X





