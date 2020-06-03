# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)
import os
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from net import Net
from dataset import Dataset
from net import Loss
from net import Optimizers
from writer import Logger
from utils import get_config
from .decoder import Decoder


np.random.seed(4)
torch.manual_seed(4)

def decode(aegs, n_spk):
    # load config
    config = get_config(args.conf_path)

    # logger
    logger = Logger(args.log_name, 'decoder', 'dataset')

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.decoder.info('device: %s' % device)

    # trainind settings and model
    net = Net(config.model, n_spk, 1, device)
    net.to(device)

    # resume
    dic = torch.load(args.checkpoint)
    net.load_state_dict(dic['model'])
    criteria_before = dic['criteria']
    iter_count = dic['iter_count']
    logger.decoder.info(net)
    logger.decoder.info('Criteria before: %f' % criteria_before)

    # dataset
    datasets = {'test': Dataset(args.test_dir, args.stats_dir,
                                 logger.dataset, pad_len=2800,
                                 batch_len=0, device=device)
                }

    data_loaders = {'test': DataLoader(datasets['test'],
                                        batch_size=1,
                                        shuffle=True)
                    }

    # logging about training data
    logger.dataset.info('number of test samples: %d' % len(datasets['test']))

    # decoder for validation
    decoder = Decoder(args.exp_dir, datasets['test'].scaler, logger=logger.decoder)

    # decode
    logger.decoder.info('start decoding!')
    for i, batch in enumerate(data_loaders['test']):
        # inputs
        inputs = {
                    'feat': torch.cat((
                                batch['uv'],
                                batch['lcf0'],
                                batch['codeap'],
                                batch['mcep']),
                            dim=-1).to(device),
                    'cv_stats': torch.cat((
                                batch['uv'],
                                batch['lcf0'],
                                batch['codeap']),
                            dim=-1).to(device),
                    'src_code': batch['src_code'].to(device),
                    'trg_code': batch['trg_code'].to(device),
                    'src_spk': batch['src_id'].to(device),
                    'trg_spk': batch['trg_id'].to(device),
                    'src': batch['src_spk'],
                    'trg': batch['trg_spk'],
                    'flen': batch['flen'],
                    'f0': batch['f0'],
                    'codeap': batch['codeap'],
                    'mcep': batch['mcep'],
                    'cv_f0': batch['cv_f0']
            }

        # forward propagation with target-pos output
        outputs = net(inputs)

        # decode
        decoder.decode(inputs, outputs, iter_count, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default=None, type=str,
                        help='Path to the training data.')
    parser.add_argument('--exp_dir', default=None, type=str,
                        help='Path to the validation data')
    parser.add_argument('--stats_dir', default=None, type=str,
                        help='Path to the stats files.')
    parser.add_argument('--conf_path', default=None, type=str,
                        help='Path to the config file')
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to the directory where trained model will be saved.')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Name log file will be saved')
    args = parser.parse_args()

    n_spk = 2

    # decode
    decode(args, n_spk=n_spk)
