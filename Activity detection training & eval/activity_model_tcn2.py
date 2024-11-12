'''
TCN model architecture, v2
Based on ModernTCN
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class PWConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

class ConvFF(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

class DWConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

class Patching(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

class AccModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples)
    '''
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.patching = Patching()
        self.dw_conv = DWConv()
        self.conv_ff = ConvFF()
        self.bn = nn.BatchNorm1d()
        self.head = nn.Linear()

    def forward(self, X):

        res = self.patching(X)          # res = input into proxy-transformer architecture

        X = self.bn(self.dw_conv(res))
        X = res + self.conv_ff(self.conv_ff(X))
        ## flatten
        out = self.head(X)

        return X






