'''
Model combining convolution along channels & cross attention across channels
'''

import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    '''
    Causal temporal convolution block
    '''
    def __init__(self, kernel_size=5, dilation_rate=2, pool_size=2, padding='causal'):

        '''
        Input shape =
        :param kernel_size:
        :param dilation_rate:
        :param pool_size:
        :param padding:
        '''

        super().__init__()






