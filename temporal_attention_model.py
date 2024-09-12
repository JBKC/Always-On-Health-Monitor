'''
Model combining convolution along channels & cross attention across channels
'''

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    '''
    Single causal temporal convolution block
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

class TemporalConvolution(nn.Module):
    '''
    Pass data through series of convolution blocks
    '''

    def __init__(self):
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=in_channels, n_filters=32, pool_size=4)
        self.conv_block2 = ConvBlock(in_channels=in_channels, n_filters=48)
        self.conv_block3 = ConvBlock(in_channels=in_channels, n_filters=64)

    def forward(self, x):
        '''
        :param x:
        :return:
        '''

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        return x



class AttentionModule(nn.Module):
    '''
    Cross-Attention module
    Input is the output of the convolutional blocks
    '''

    def __init__(self, embed_dim=16, num_heads=4):
        super().__init__()

        # single cross-attention module
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        '''
        :param query:
        :param key:
        :param value:
        :return:
        '''

        out, _ = self.cross_attention(query, key, value)

        return out


class TemporalAttentionModel(nn.Module):
    '''
    Full architecture build
    '''

    def __init__(self):
        super().__init__()

        self.attention = AttentionModule()
        self.ln = nn.LayerNorm()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.dropout = nn.Dropout(p=0.125)


    def forward(self, input):
        '''
        Input shape =
        :param input:
        :return:
        '''

        # SPLIT INPUT INTO X_BVPi & X_BVP_i-1s

        x = self.attention(x)

        x = self.fc1(self.ln(x))
        x = self.fc2(self.dropout(x))

        return x

    def loss_func(self):
        return















