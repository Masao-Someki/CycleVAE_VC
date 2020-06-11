# training script

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import librosa

from dataset import VocoderDataset
from writer import Logger
from net import Optimizers
from utils import get_config
from net import VocoderNet
from loss import GenLoss, DisLoss

np.random.seed(4)
torch.manual_seed(4)

def train(args, n_spk):
    # load config
    config = get_config(args.conf_path)

    # logger
    logger = Logger(args.log_name, 'decoder', 'dataset')

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-set_parameters
    batch_len = 80
    hop_size = 120

    # trainind settings and model
    net = VocoderNet(config.model, upsample_params={"upsample_scales": [4, 5, 6]}, device=device)
    net.to(device)
    iter_count = 0
    GenOptim = Optimizers(config.gen_optim)
    GenOptim.set_parameters(list(net.generator.named_parameters()))
    DisOptim = Optimizers(config.dis_optim)
    DisOptim.set_parameters(list(net.discriminator.named_parameters()))
    criteria_before = 10000
    past_model = ''

    # load checkpoint
    logger.decoder.info('Loading checkpoint from %s' % args.checkpoint)
    dic = torch.load(args.checkpoint)
    net.load_state_dict(dic['model'])
    iter_count = dic['iter_count']
    GenOptim = dic['GenOptim']
    DisOptim = dic['DisOptim']
    criteria_before = dic['criteria']
    past_model = dic['path']

    # dataset
    datasets = {'decoder': VocoderDataset(args.decode_dir, args.stats_dir,
                                 logger.dataset, hop_size=hop_size, batch_len=-1,
                                 device=device)
                }

    data_loaders = {'decoder': DataLoader(datasets['decoder'],
                                        batch_size=1,
                                        shuffle=False)}

    # log
    # logging about training data
    logger.dataset.info('number of decoding samples: %d' % len(datasets['decoder']))
    logger.decoder.info(net)
    logger.decoder.info('Start training from iteration %d' % iter_count)

    # directory
    wav_dir = os.path.join(args.exp_dir, str(iter_count))
    if not os.path.exists(wav_dir):
        os.mkdir(wav_dir)

    # train PWG
    count = 0
    for batch in data_loaders['decoder']:
        count += 1
        logger.decoder.info('Number of processing data: %d / %d' \
                % (count, len(datasets['decoder'])))

        # Compute over each data
        wav = np.zeros(batch['cvwav'].shape[-1])
        n_mini = batch['cvmcep'].shape[-1] // batch_len

        for i in range(n_mini):
            # slice
            start = i * batch_len
            end = start + batch_len
            start_wav = start * hop_size
            end_wav = end * hop_size

            # data to device
            inputs = {
                    'cvmcep': batch['cvmcep'][..., start:end].to(device),
                    'wav': batch['wav'][..., start_wav:end_wav].to(device),
                    'cvwav': batch['cvwav'][..., start_wav:end_wav].to(device)
            }

            # if
            #######################
            ###### Generator ######
            #######################
            # forward propagation
            out_gen = net.generator(inputs)

            # save wav file
            wav_file = os.path.join(
                        wav_dir,
                        '%s_%s_%d.wav' % ('jvs001', 'yukari', i)
            )
            wav_part = out_gen['reconst'][0][0].cpu().detach().numpy()
            wav[start_wav:end_wav] = wav_part

        librosa.output.write_wav(wav_file, wav, 24000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode_dir', default=None, type=str,
                        help='Path to the training data.')
    parser.add_argument('--exp_dir', default=None, type=str,
                        help='Path to the validation data')
    parser.add_argument('--stats_dir', default=None, type=str,
                        help='Path to the stats files.')
    parser.add_argument('--conf_path', default=None, type=str,
                        help='Path to the config file')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to the checkpoint model file')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Name log file will be saved')
    args = parser.parse_args()

    # number pf speakers
    n_spk = 2

    # train
    train(args, n_spk=n_spk)
