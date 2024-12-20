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

        zl = self.bn(self.dw_large(z))
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

class Backbone(nn.Module):
    '''
    input shape = (B,M,D,N)
    '''
    def __init__(self, M, D):
        super().__init__()

        self.dw = DWConv(M,D)
        self.bn = nn.BatchNorm1d(D)
        self.ff1 = ConvFFN1(M,D)
        self.ff2 = ConvFFN2(M,D)

    def forward(self, x):

        B, M, D, N = x.shape

        # reshape for DWConv
        z = x.reshape(B, M * D, N)
        z = self.dw(z)

        # apply batchnorm over embedding dimension
        z = z.reshape(B, M, D, N)
        z = z.reshape(B * M, D, N)
        z = self.bn(z)
        z = z.reshape(B, M, D, N)
        z = z.reshape(B, M * D, N)

        # ConvFFN1
        z = self.ff1(z)

        # reshape & permute for ConvFFN2
        z = z.reshape(B, M, D, N)
        z = z.permute(0, 2, 1, 3)
        z = z.reshape(B, D * M, N)
        z = self.ff2(z)

        # reshape & permute to match input
        z = z.reshape(B, D, M, N)
        z = z.permute(0, 2, 1, 3)

        out = x + z

        return out


class AccModel(nn.Module):
    '''
    Full Model architecture
    input shape = (batch_size, n_channels, n_samples) = (B,M,L)
    '''
    def __init__(self, in_channels, num_classes, n_blocks=2):
        super().__init__()

        dmin = 32
        dmax = 512
        n_embd = min(max(2 ** np.log(in_channels), dmin), dmax)     # embedding dimension (D)

        self.patching = Patching(n_embd)
        self.backbone = nn.ModuleList([Backbone(M=in_channels,D=n_embd) for _ in range(n_blocks)])
        self.head = nn.Linear(in_channels*n_embd*256, num_classes)

    def forward(self, x):

        B,M,L = x.shape

        # reshape for patching
        x = x.unsqueeze(2)
        x = x.reshape(B*M, 1, L)
        x = self.patching(x)

        # reshape for backbone input
        x = x.reshape(B, M, x.size(1), x.size(-1))       # acts as input into model

        for block in self.backbone:
            x = block(x)

        # pass into Head
        x = x.view(x.size(0), -1)
        out = self.head(x)

        return out






