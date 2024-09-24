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

    def __init__(self, in_channels, n_filters, pool_size, kernel_size=5, dilation=2, dropout=0.5):

        super().__init__()

    def forward(self, x):
        return




class AccModel(nn.Module):
    '''
    Model architecture that takes only accelerometer channels
    '''
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return




