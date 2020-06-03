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
        self.ar = False

        # receptive_field
        rec_field = self.config.dil_conv.kernel_size ** self.config.dil_conv.n_convs
        self.padding = nn.ConstantPad1d(int((rec_field - 1) / 2), 0)

        # DilConv
        self.dil_conv = DilConv(self.config.dil_conv)

        # Rnn
        self.rnn = RNN(self.config.rnn, self.config.dil_conv, device=self.device)
        self.h_size = self.rnn.h_size #, self.rnn2.h_size]

        # out_dim
        self.conv = nn.Conv1d(
                self.config.rnn.h_units * (self.config.rnn.bidirectional + 1),
                self.config.rnn.out_dim, 1)

        # AR model
        if self.config.rnn.model_arch == 'ar':
            self.rnn.add_conv(self.conv)
            self.ar = True
        elif self.config.rnn.model_arch == "rnn":
            self.conv2 = nn.Conv1d(self.rnn.in_dim + self.config.rnn.out_dim,
                    self.config.rnn.out_dim, 1)
        else:
            raise ValueError('model arch %s is not supported.' % self.config.rnn.model_arch)

    def forward(self, x):
        # shape of x: (B, L, D)
        # add padding

        x = x.transpose(1, 2) # (B, D, L)
        x = self.padding(x) # (B, L, D + padding)
        x = self.dil_conv(x) # (B, D, L)
        conv_out = x.transpose(1, 2) # (B, L, D)-

        x = conv_out.transpose(0, 1) # (L, B, D)

        # rnn-based or ar based.
        x = self.rnn(x) # (L, B, D)
        x = x.transpose(0, 1) # (B, L, D)

        if not self.ar:
            x = x.transpose(1, 2) # (B, L, D) to (B, D, L)
            x = self.conv(x) # (B, D, L)
            x = x.transpose(1, 2) # (B, D, L) to (B, L, D)

            x = torch.cat((x, conv_out), 2) #( B, L, D)
            x = x.transpose(1, 2)
            x = self.conv2(x)
            x = x.transpose(1, 2) # B, L, D)

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
        self.ar = False

        self.in_dim = config_conv.in_dim * config_conv.kernel_size ** config_conv.n_convs
        self.set_rnn_layer()

        self.dropout = nn.Dropout(p=0.5)
        self.h_size = (self.config.n_layers * (self.config.bidirectional + 1), 
                1, self.config.h_units)

    def add_conv(self, layer):
        self.ar_layer = layer
        self.in_dim = self.in_dim + self.config.out_dim
        self.set_rnn_layer()
        self.ar = True

    def set_rnn_layer(self):
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

    def forward(self, x):
        # x: (L, B, D)
        if self.ar:
            y_in = torch.zeros((1, x.shape[1], self.config.out_dim)).to(self.device)
            h_in = torch.zeros((self.config.n_layers * (self.config.bidirectional + 1),
                                x.shape[1], self.config.h_units)).to(self.device)

            # AR model
            ret = torch.Tensor([]).to(self.device)
            for i in range(x.shape[0]):
                inputs = torch.cat((y_in, x[i:(i+1)]), dim=2)
                out, h_in = self.rnn(inputs, h_in) # (L, B, D)
                out = self.ar_layer(out.transpose(0, 1).transpose(1, 2)) # (B, D, L)
                y_in = out.transpose(1, 2).transpose(0, 1) # (L, B, D)
                ret = torch.cat((ret, y_in), dim=0) # (L, B, D)
        else:
            ret,_ = self.rnn(x)
            ret = self.dropout(ret)
        return ret
