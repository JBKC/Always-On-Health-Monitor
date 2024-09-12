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



class TemporalAttentionModel(nn.Module):
    '''
    Full architecture build
    '''

    def __init__(self):
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=in_channels, n_filters=32, pool_size=4)
        self.conv_block2 = ConvBlock(in_channels=in_channels, n_filters=48)
        self.conv_block3 = ConvBlock(in_channels=in_channels, n_filters=64)
        self.attention = AttentionModule()
        self.ln = nn.LayerNorm()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.dropout = nn.Dropout(p=0.125)


    def forward(self, input):

        # SPLIT INPUT INTO X_BVPi & X_BVP_i-1s

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.attention(x)

        x = self.fc1(self.ln(x))
        x = self.fc2(self.dropout(x))

        return x

    def loss_func(self):
        return















