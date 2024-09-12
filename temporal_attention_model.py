'''
Model combining convolution of time window batches & attention across adjacent time window pairs
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

    def __init__(self, in_channels=256):
        super().__init__()

        self.conv_block1 = ConvBlock(in_channels=in_channels, n_filters=32, pool_size=4)
        self.conv_block2 = ConvBlock(in_channels=32, n_filters=48)
        self.conv_block3 = ConvBlock(in_channels=48, n_filters=64)

    def forward(self, x_cur, x_prev):
        '''
        Pass both x_cur and x_prev through the same convolution blocks (weight sharing)
        :param x_cur:
        :return:
        '''

        x_cur = self.conv_block1(x_cur)
        x_prev = self.conv_block1(x_prev)

        x_cur = self.conv_block2(x_cur)
        x_prev = self.conv_block2(x_prev)

        x_cur = self.conv_block3(x_cur)
        x_prev = self.conv_block3(x_prev)

        return x_cur, x_prev



class AttentionModule(nn.Module):
    '''
    Cross-Attention module
    Input is the output of the convolutional blocks
    '''

    def __init__(self, embed_dim=16, num_heads=4):
        super().__init__()

        # single attention module
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        '''
        :param query:
        :param key:
        :param value:
        :return:
        '''


        out, _ = self.attention(query, key, value)

        return out


class TemporalAttentionModel(nn.Module):
    '''
    Attention architecture build
    '''

    def __init__(self):
        super().__init__()

        self.convolution = TemporalConvolution()
        self.attention = AttentionModule()
        # self.ln = nn.LayerNorm()
        # self.fc1 = nn.Linear(256)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.125)
        # self.fc2 = nn.Linear(2)

    def forward(self, x_cur, x_prev):
        '''
        :param x_cur:
        :param x_prev:
        :return:
        '''

        # temporal convolution
        x_cur, x_prev = self.convolution(x_cur, x_prev)

        # attention with residual connection: query = x_prev, key = value = x_cur
        x = x_cur + self.attention(x_prev, x_cur, x_cur)

        x_cur = x_cur.flatten()

        x = self.relu(self.fc1(self.ln(x)))
        x = self.fc2(self.dropout(x))

        return x

    def loss_func(self):
        return















