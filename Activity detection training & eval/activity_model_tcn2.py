'''
TCN model architecture, v2
Based on ModernTCN
'''

import numpy as np
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
    '''
    input shape = (B,M*D,N)
    '''
    def __init__(self):
        super().__init__()

    def forward(self, z):

        print(z.shape)

        

        return

class Patching(nn.Module):
    '''
    input shape = (B,M,1,L)
    '''
    def __init__(self, n_embd, patch_size=1, stride=1):
        super().__init__()

        self.patch_conv = nn.Conv1d(in_channels=1,out_channels=n_embd,kernel_size=patch_size,stride=stride,
                                    padding=(patch_size-1)//2 if patch_size % 2 == 0 else patch_size//2)

    def forward(self, x):

        # reshape for Pytorch convolution (B*M,1,L)
        x = x.view(-1, 1, x.size(-1))
        x = self.patch_conv(x)

        return x

class AccModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples) = (B,M,L)
    '''
    def __init__(self, in_channels):
        super().__init__()

        dmin = 32
        dmax = 512
        n_embd = min(max(2 ** np.log(in_channels), dmin), dmax)

        self.patching = Patching(n_embd)
        self.dw_conv = DWConv()
        self.conv_ff = ConvFF()
        self.bn = nn.BatchNorm1d(n_embd)
        # self.head = nn.Linear()

    def forward(self, x):

        B,M,L = x.shape

        # reshape for patching
        x = x.unsqueeze(2)
        res = self.patching(x)       # res = input into proxy-transformer architecture
        print(res.shape)

        # reshape for DWConv
        z = res.view(B, M*res.size(1), res.size(-1))
        z = self.bn(self.dw_conv(z))


        #
        # x = res + self.conv_ff(self.conv_ff(x))
        # ## flatten
        # out = self.head(x)

        return out






