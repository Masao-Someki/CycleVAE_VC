# This script is the model definitioimportn

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from net import ConvRNN

def sampling(x, device):
    # shape of x: (B, L, D) where B: Batch_size, L: length, D: latend_dim * 2
    lat_dim = x.shape[2] // 2

    mu = x[:, :, :lat_dim]
    sigma = x[:, :, lat_dim:]
    eps = torch.randn(x.shape[0], x.shape[1], lat_dim).to(device)
    return mu + torch.exp(sigma / 2) * eps

class Net(nn.Module):
    def __init__(self, config, n_spk, n_cyc=2, device=None):
        super(Net, self).__init__()
        self.config = config
        self.n_spk = n_spk
        self.device = device
        self.n_cyc = n_cyc

        # HyperParameter
        self.clamp_min = -1 * np.log(1000000)

        # encoder
        self.enc = ConvRNN(self.config.encoder, device)

        # decoder
        self.dec = ConvRNN(self.config.decoder, device)

    def forward(self, x):
        output = {}
        output['latent_1'] = []
        output['latent_2'] = []
        output['reconst_half'] = []
        output['reconst_last'] = []

        # cycle
        encin = x['feat']

        for c in range(self.n_cyc):
            ##################
            ### First half ###
            ##################
            # Encoder
            # (B, L, D_in) to (B, L, n_spk + D_lat*2)
            encoded = self.enc(encin)

            # latent dim
            lat_dim = encoded.shape[2] // 2# + self.n_spk
            encoded = torch.cat((
                encoded[:,:,:lat_dim],
                torch.clamp(encoded[:,:,lat_dim:],min=self.clamp_min)),
            2)

            # sampling
            output['latent_1'].append(encoded)
            sampled = sampling(encoded, self.device)

            # Decoder trg
            # (B, L, n_spk + lat_dim)
            decin_trg = torch.cat((x['trg_code'], sampled), dim=2)
            dec_trg = self.dec(decin_trg) # (B, L, D_out)
            output['trg_reconst'] = dec_trg

            # Decoder src
            decin_src = torch.cat((x['src_code'], sampled), dim=2)

            # (B, L, n_spk + lat_dim)
            dec_src = self.dec(decin_src) # (B, L, D_out)
            output['reconst_half'].append(dec_src)

            ###################
            ### Second half ###
            ###################

            # Encoder
            encin = torch.cat((x['cv_stats'], dec_trg), dim=2)
            encoded = self.enc(encin) # (B, L, D)

            # latent dim
            lat_dim = encoded.shape[-1] // 2# + self.n_spk
            encoded = torch.cat((
                encoded[:,:,:lat_dim],
                torch.clamp(encoded[:,:,lat_dim:],min=self.clamp_min)),
            2)

            # sampling
            output['latent_2'].append(encoded)
            sampled = sampling(encoded, self.device)

            # Decoder src
            # (B, L, n_spk + lat_dim)
            decin_src = torch.cat((x['src_code'], sampled), dim=2)
            dec_src = self.dec(decin_src) # (B, L, D_out)
            output['reconst_last'].append(dec_src)

            # set encin for the next cycle
            encin = torch.cat((x['cv_stats'], dec_src), dim=2)

        return output
