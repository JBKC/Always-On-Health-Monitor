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

    def __init__(self, in_channels, n_filters, pooling_size, stride, kernel_size=[11,21,41]):

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

        self.bn = nn.BatchNorm1d(n_filters)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, X):

        # pass through bottleneck
        X1x1 = F.relu(self.bn(self.conv1x1(X)))
        X1x1 = self.dropout(X1x1)

        # pass through convolution branches in parallel to get list
        conv_out = [self.bn(branch(X1x1)) for branch in self.branches]

        # pass through maxpool branch
        max_out = self.bn(self.conv1x1(self.pooling(X)))
        conv_out.append(max_out)

        # concatenate outputs across channel dimension
        out = F.relu(torch.cat(conv_out, dim=1))
        out = self.dropout(out)

        return out


class ConvBlocks(nn.Module):
    '''
    Series of repeating Inception blocks
    '''

    def __init__(self):

        super().__init__()

        in_channels = 3
        n_filters = 16
        n_blocks = 6
        stride = 1
        pooling_size = 3
        n_out = n_filters*4

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

        # save input for residual connection
        residual = X

        # iterate over each block in the ModuleList
        for i, block in enumerate(self.conv_blocks):

            X = block(X)

            # residual block every 3rd connection
            if (i + 1) % 3 == 0:
                if residual.shape != X.shape:           # shape mismatch will occur with the first residual
                    residual = self.bn(self.res1x1(residual))

                X = F.relu(self.bn(X + residual))
                residual = X

        return X


class AccModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples)
    '''
    def __init__(self):
        super().__init__()

        n_activities = 8

        self.conv = ConvBlocks()
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)  # global average pooling

        self.fc = nn.Linear(32, n_activities)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, X):

        # pass through inception time blocks
        X = self.conv(X)

        # perform global average pooling
        X = torch.squeeze(self.gap(X), dim=-1)
        X = self.dropout(X)

        # FCN to output
        X = self.fc(X)

        return X






