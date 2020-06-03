# Script for ConvRNN

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConvRNN(nn.Module):
    def __init__(self, model_config, device):
        super(ConvRNN, self).__init__()
        self.config = model_config
        self.device = device

        # receptive_field
        rec_field = self.config.dil_conv.kernel_size ** self.config.dil_conv.n_convs
        self.padding = nn.ConstantPad1d(int((rec_field - 1) / 2), 0)

        # DilConv
        self.dil_conv = DilConv(self.config.dil_conv)

        # Rnn
        self.rnn = RNN(self.config.rnn, self.config.dil_conv, device=self.device)
        self.h_size = self.rnn.h_size #, self.rnn2.h_size]

        # out_dim
        if self.config.rnn.bidirectional:
            self.conv = nn.Conv1d(self.config.rnn.h_units * 2, self.config.rnn.out_dim, 1)
            self.conv2 = nn.Conv1d(self.rnn.in_dim + self.config.rnn.out_dim,
                    self.config.rnn.out_dim, 1)
        else:
            self.conv = nn.Conv1d(self.config.rnn.h_units, self.config.rnn.out_dim, 1)
            self.conv2 = nn.Conv1d(self.rnn.in_dim + self.config.rnn.out_dim,
                    self.config.rnn.out_dim, 1)

    def forward(self, x):
        # shape of x: (B, L, D)
        # add padding

        x = x.transpose(1, 2) # (B, D, L)
        x = self.padding(x) # (B, L, D + padding)
        x = self.dil_conv(x) # (B, D, L)
        conv_out = x.transpose(1, 2) # (B, L, D)-

        x = conv_out.transpose(0, 1) # (L, B, D)
        x = self.rnn(x) # (L, B, D)
        x = x.transpose(0, 1) # (B, L, D)

        x = x.transpose(1, 2) # (B, L, D) to (B, D, L)
        x = self.conv(x) # (B, D, L)
        x = x.transpose(1, 2) # (B, D, L) to (B, L, D)

        x = torch.cat((x, conv_out), 2) #( B, L, D)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)

        return x


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dilation):
        super(GLU, self).__init__()
        self.out_dim = out_dim

        self.conv = nn.Conv1d(in_dim, out_dim * 2, kernel_size, dilation=dilation)

    def forward(self, x):
        x = self.conv(x) # (B, D, L) to (B, D*2, L)
        x_base = x[:, :self.out_dim]
        x_sigma = x[:, self.out_dim:]

        return x_base * torch.sigmoid(x_sigma)


class DilConv(nn.Module):
    def __init__(self, config):
        super(DilConv, self).__init__()
        self.config = config

        # DilConv
        self.convs = nn.ModuleList()

        for i in range(self.config.n_convs):
            if self.config.conv_type == 'glu':
                # Gated Linear Units
                self.convs += [
                        GLU(self.config.in_dim * self.config.kernel_size ** i,
                            self.config.in_dim * self.config.kernel_size ** (i + 1),
                            self.config.kernel_size,
                            self.config.kernel_size ** i
                        )
                ]
            elif self.config.conv_type == 'linear':
                # Normal conv1d
                self.convs += [
                        nn.Conv1d(
                            self.config.in_dim * self.config.kernel_size ** i,
                            self.config.in_dim * self.config.kernel_size ** (i + 1),
                            self.config.kernel_size,
                            dilation=self.config.kernel_size ** i
                        )
                ]
            else:
                raise ValueError('conv type %s is not supported now.' % self.config.conv_type)

        # dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        for i, layer in enumerate(self.convs):
            x = layer(x)
        x = self.dropout(x)
        return x


class RNN(nn.Module):
    def __init__(self, config_rnn, config_conv, device=None):
        super(RNN, self).__init__()
        self.config = config_rnn
        self.device = device

        self.in_dim = config_conv.in_dim * config_conv.kernel_size ** config_conv.n_convs
        if self.config.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                    self.in_dim,
                    hidden_size=self.config.h_units,
                    num_layers=self.config.n_layers,
                    bidirectional=bool(self.config.bidirectional)
            )
        elif self.config.rnn_type == 'gru':
            self.rnn = nn.GRU(
                    self.in_dim,
                    hidden_size=self.config.h_units,
                    num_layers=self.config.n_layers,
                    bidirectional=bool(self.config.bidirectional)
            )
        else:
            raise ValueError('rnn type %s is not supported.' % self.config.rnn_type)

        self.dropout = nn.Dropout(p=0.5)
        self.h_size = (self.config.n_layers * (self.config.bidirectional + 1), 1, self.config.h_units)

    def forward(self, x):
        ret, _ = self.rnn(x)
        ret = self.dropout(ret)
        return ret
