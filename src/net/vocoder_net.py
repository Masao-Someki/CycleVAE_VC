# This script is the model definitioimportn

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from parallel_wavegan.models import ParallelWaveGANGenerator
from parallel_wavegan.models import ParallelWaveGANDiscriminator


class VocoderNet(nn.Module):
    def __init__(self, config, upsample_params, device=None):
        super(VocoderNet, self).__init__()
        self.config = config
        self.device = device

        # net
        self.generator = Generator(self.config.generator, upsample_params)
        self.discriminator = Discriminator(self.config.discriminator.dic)

class Generator(nn.Module):
    def __init__(self, config, upsample_params):
        super(Generator, self).__init__()
        self.net = ParallelWaveGANGenerator(**config, upsample_params=upsample_params)

    def forward(self, inputs):
        # compute pwg
        return {'reconst': self.net(inputs['cvwav'], inputs['cvmcep'])}

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.net = ParallelWaveGANDiscriminator(**config)

    def forward(self, inputs, out_gen):
        return {
            'real': self.net(inputs['wav']),
            'fake': self.net(out_gen['reconst'].detach())
        }
