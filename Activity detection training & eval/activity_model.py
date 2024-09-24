'''
Model architecture for activity detection
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal



class ConvBlock(nn.Module):
    '''
    Single convolution block
    '''

    def __init__(self, in_channels, n_filters, pool_size, kernel_size=5, dropout=0.5):

        super().__init__()

        self.kernel_size = kernel_size
        self.n_filters = n_filters

        # conv block = conv layer + BN + ReLu + pooling
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                ),
                nn.ReLU()
            ) for i in range(3)]
        )

        self.pool = nn.AvgPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):


        return




class AccModel(nn.Module):
    '''
    Model architecture that takes only accelerometer channels
    input shape = (batch_size, n_channels, n_samples)
    '''
    def __init__(self):
        super().__init__()

        self.convolution = ConvBlock()




    def forward(self, x):
        return




