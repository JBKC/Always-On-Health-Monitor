'''
Model combining convolution along channels & cross attention across channels
'''

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    '''
    Causal temporal convolution block
    Each block contains 3 convolutional layers
    '''

    def __init__(self, in_channels, n_filters, kernel_size=5, dilation=2, pool_size=2, dropout=0.5):

        super().__init__()

        # causal padding
        padding = dilation * (kernel_size - 1)

        # conv block = [conv layer + RELU] * 3 + average pooling + dropout
        self.conv = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(
                    in_channels if i == 0 else n_filters,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding
                ),
                nn.ReLU()
            ) for i in range(3)],
            nn.AvgPool1d(kernel_size=pool_size),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        '''
        Pass data through the convolution block
        :param x:
        :return:
        '''

        return self.conv(x)


class AttentionModule(nn.Module):
    '''
    Cross-Attention module
    Input is the output of the convolutional blocks
    '''

    def __init__(self):
        super().__init__()


















