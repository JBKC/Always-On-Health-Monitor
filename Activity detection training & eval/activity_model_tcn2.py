'''
TCN model architecture, v2
Based on ModernTCN
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFFN1(nn.Module):
    '''
    input shape = (B,M*D,N)
    '''
    def __init__(self, M, D):
        super().__init__()

        r = 1
        self.pw11 = nn.Conv1d(in_channels=M * D, out_channels=r * M * D, kernel_size=1, stride=1, groups=M)
        self.pw12 = nn.Conv1d(in_channels=r * M * D, out_channels=M * D, kernel_size=1, stride=1, groups=M)

    def forward(self, z):

        z = self.pw12(F.gelu(self.pw11(z)))

        ## dropout goes here

        return z

class ConvFFN2(nn.Module):
    '''
    input shape = (B,D*M,N)
    '''
    def __init__(self, M, D):
        super().__init__()

        r = 1
        self.pw21 = nn.Conv1d(in_channels=D * M, out_channels=r * D * M, kernel_size=1, stride=1, groups=D)
        self.pw22 = nn.Conv1d(in_channels=r * D * M, out_channels=D * M, kernel_size=1, stride=1, groups=D)

    def forward(self, z):

        z = self.pw22(F.gelu(self.pw21(z)))

        ## dropout goes here

        return z

class DWConv(nn.Module):
    '''
    input shape = (B,M*D,N)
    '''
    def __init__(self, M, D, klarge=51, ksmall=5, stride=1):
        super().__init__()

        self.dw_large = nn.Conv1d(in_channels=M*D,out_channels=M*D,kernel_size=klarge,stride=stride,
                                    padding=klarge//2,groups=M*D)
        self.dw_small = nn.Conv1d(in_channels=M*D,out_channels=M*D,kernel_size=ksmall,stride=stride,
                                    padding=ksmall//2,groups=M*D)

        self.bn = nn.BatchNorm1d(M*D)

    def forward(self, z):

        zl = self.dw_large(z)
        zs = self.bn(self.dw_small(z))

        return zl+zs

class Patching(nn.Module):
    '''
    input shape = (B*M,1,L)
    '''
    def __init__(self, n_embd, patch_size=1, stride=1):
        super().__init__()

        self.patch_conv = nn.Conv1d(in_channels=1,out_channels=n_embd,kernel_size=patch_size,stride=stride,
                                    padding=patch_size//2)

    def forward(self, x):

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
        n_embd = min(max(2 ** np.log(in_channels), dmin), dmax)     # embedding dimension (D)

        self.patching = Patching(n_embd)
        self.dw = DWConv(M=in_channels,D=n_embd)
        self.bn = nn.BatchNorm1d(in_channels*n_embd)
        self.ff1 = ConvFFN1(M=in_channels,D=n_embd)
        self.ff2 = ConvFFN2(M=in_channels, D=n_embd)
        # self.head = nn.Linear()

    def forward(self, x):

        B,M,L = x.shape

        # reshape for patching
        x = x.unsqueeze(2)
        x = x.reshape(B*M, 1, L)
        x = self.patching(x)

        # reshape for backbone input
        x = x.reshape(B, M, x.size(1), x.size(-1))       # acts as input into model
        B,M,D,N = x.shape

        # reshape for DWConv
        z = x.reshape(B,M*D,N)
        z = self.dw(z)

        # apply batchnorm
        z = z.reshape(B,M,D,N)
        z = z.reshape(B*M,D,N)
        z = self.bn(z)
        z = z.reshape(B,M,D,N)
        z = z.reshape(B,M*D,N)

        # ConvFFN1
        z = self.ff1(z)

        # reshape & permute for ConvFFN2
        z = z.reshape(B, M, D, N)
        z = z.permute(0,2,1,3)
        z = z.reshape(B, D*M, N)
        z = self.ff2(z)

        # reshape & permute to match input
        z = z.reshape(B, D, M, N)
        z = z.permute(0,2,1,3)
        out = x + z
        print(out.shape)

        # pass into


        # x = res + self.conv_ff(self.conv_ff(x))
        # ## flatten
        # out = self.head(x)

        return






