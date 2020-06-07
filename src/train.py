# training script

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from net import Net
from dataset import Dataset
from loss import Loss
from net import Optimizers
from writer import Logger
from utils import get_config
from decode import Decoder

np.random.seed(4)
torch.manual_seed(4)

def train(args, n_spk):
    # load config
    config = get_config(args.conf_path)

    # logger
    logger = Logger(args.log_name, 'train', 'val', 'dataset', 'decoder')

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-set_parameters
    pad_len = 2800
    batch_len = 80
    assert pad_len % batch_len == 0

    # trainind settings and model
    net = Net(config.model, n_spk, n_cyc=2, device=device)
    net.to(device)
    iter_count = 0
    optim = Optimizers(config.optim)
    optim.set_parameters(list(net.named_parameters()))
    criteria_before = 10000
    past_model = ''

    # resume
    if args.resume is not None:
        dic = torch.load(args.resume)
        net.load_state_dict(dic['model'])
        iter_count = dic['iter_count']
        optim = dic['optim']
        criteria_before = dic['criteria']
        past_model = dic['path']

    # dataset
    datasets = {'train': Dataset(args.train_dir, args.stats_dir,
                                 logger.dataset, pad_len=pad_len,
                                 batch_len=batch_len, device=device),
                'val'  : Dataset(args.val_dir, args.stats_dir,
                                 logger.dataset, pad_len=pad_len,
                                 batch_len=batch_len, device=device)
                }

    data_loaders = {'train': DataLoader(datasets['train'],
                                        batch_size=config.train.batch_size,
                                        shuffle=True),
                    'val'  : DataLoader(datasets['val'],
                                        batch_size=config.val.batch_size,
                                        shuffle=False)}
    # logging about training data
    logger.dataset.info('number of training samples: %d' % len(datasets['train']))
    logger.dataset.info('number of validation samples: %d' % len(datasets['val']))

    # loss function
    loss_fn = Loss(device)

    # log net
    logger.train.info(net)

    # training!
    logger.train.info('Start training from iteration %d' % iter_count)

    # Hypre-parameter required to compute loss
    scale_var = torch.Tensor(datasets['train'].scaler['mcep'].scale_).to(device)

    # train
    for e in range(config.train.epoch):
        net.train()
        losses = []

        for batch in data_loaders['train']:
            # iter for batch
            iter_count += 1

            # training data to device
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
                    'flen': batch['flen'].to(device)
            }

            # forward propagation
            out = net(inputs)

            # compute loss
            loss, loss_dic = loss_fn(out, inputs, scale_var)

            # backward
            loss.backward()
            optim.step()

            # log
            losses.append(loss.cpu().detach().numpy())

            if iter_count % config.train.log_every == 0:
                logger.train.info('loss at iter %d : %s' % (iter_count, loss.item()))
                logger.train.figure('train', loss_dic, iter_count)
        # log
        logger.train.info('Loss for epoch %d : %.5f' % (e, np.mean(losses)))

        # Validation
        logger.val.info('Start validation at epoch %d' % e)
        net.eval()
        losses = []
        criterias = []
        with torch.no_grad():
            for batch in data_loaders['val']:
                # to device
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
                    'flen': batch['flen'].to(device)
                }

                # forward propagation
                out = net(inputs)

                # compute loss
                loss, loss_dic = loss_fn(out, inputs, scale_var)
                criteria = loss_dic['mcd/1st']

                # log
                losses.append(loss.cpu().detach().numpy())
                criterias.append(criteria)

        # log
        criteria = np.mean(criterias)
        logger.val.info('Validation loss at epoch %d: %.5f' % (e, np.mean(losses)))
        logger.val.info('Validation criteria: %.5f' % criteria)
        logger.val.figure('val', loss_dic, iter_count)

        # save model with best criteria
        if criteria < criteria_before:
            logger.val.info('Passed criteria (%f < %f), saving best model...' \
                % (criteria, criteria_before))

            # remove the existing past model.
            if not past_model == '':
                os.remove(past_model)
                logger.val.info('Found existing model at %s. Removed this file.' % past_model)

            # build dict
            save_file = os.path.join(args.model_dir, '%s.%d.pt' % (args.model_name, iter_count))
            save_dic = {
                    'model': net.state_dict(),
                    'iter_count': iter_count,
                    'optim': optim,
                    'criteria': criteria,
                    'path': save_file
                }
            torch.save(save_dic, save_file)
            criteria_before = criteria
            past_model = save_file

    # close SummaryWriter
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default=None, type=str,
                        help='Path to the training data.')
    parser.add_argument('--val_dir', default=None, type=str,
                        help='Path to the validation data')
    parser.add_argument('--stats_dir', default=None, type=str,
                        help='Path to the stats files.')
    parser.add_argument('--total_stats', default=None, type=str,
                        help='Path to the total_stats.h5')
    parser.add_argument('--conf_path', default=None, type=str,
                        help='Path to the config file')
    parser.add_argument('--model_dir', default=None, type=str,
                        help='Path to the directory where trained model will be saved.')
    parser.add_argument('--model_name', default=None, type=str,
                        help='Model name.')
    parser.add_argument('--decode_dir', default=None, type=str,
                        help='Path to the directory where decoded wav files will be saved')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Name log file will be saved')
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to the checkpoint file.')
    args = parser.parse_args()

    # number pf speakers
    n_spk = 2

    # train
    train(args, n_spk=n_spk)
