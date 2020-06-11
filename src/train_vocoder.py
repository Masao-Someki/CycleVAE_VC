# training script

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    logger = Logger(args.log_name, 'train', 'val', 'dataset', 'decoder')

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-set_parameters
    batch_len = 80

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

    # resume
    if args.resume is not None:
        logger.train.info('Loading checkpoint from %s' % args.resume)
        dic = torch.load(args.resume)
        net.load_state_dict(dic['model'])
        iter_count = dic['iter_count']
        GenOptim = dic['GenOptim']
        DisOptim = dic['DisOptim']
        criteria_before = dic['criteria']
        past_model = dic['path']

    # dataset
    datasets = {'train': VocoderDataset(args.train_dir, args.stats_dir,
                                 logger.dataset, hop_size=120,
                                 device=device),
                'val'  : VocoderDataset(args.val_dir, args.stats_dir,
                                 logger.dataset, hop_size=120,
                                 device=device)
                }

    data_loaders = {'train': DataLoader(datasets['train'],
                                        batch_size=config.train.batch_size,
                                        shuffle=True),
                    'val'  : DataLoader(datasets['val'],
                                        batch_size=config.val.batch_size,
                                        shuffle=False)}

    # loss function
    loss_gen = GenLoss().to(device)
    loss_dis = DisLoss().to(device)

    # log
    # logging about training data
    logger.dataset.info('number of training samples: %d' % len(datasets['train']))
    logger.dataset.info('number of validation samples: %d' % len(datasets['val']))
    logger.train.info(net)
    logger.train.info('Start training from iteration %d' % iter_count)

    # train PWG
    for e in range(config.train.epoch):
        net.train()
        gen_losses = []
        dis_losses = []

        for batch in data_loaders['train']:
            # iter for batch
            iter_count += 1

            # training data to device
            inputs = {
                    'cvmcep': batch['cvmcep'].to(device),
                    'wav': batch['wav'].to(device),
                    'cvwav': batch['cvwav'].to(device)
            }

            #######################
            ###### Generator ######
            #######################
            # forward propagation
            out_gen = net.generator(inputs)

            # compute loss
            if iter_count < config.train.discriminator_start:
                gen_loss, gen_loss_dic = loss_gen(out_gen, inputs)
            else:
                gen_loss, gen_loss_dic = loss_gen(out_gen, inputs, net.discriminator)

            # backward
            gen_loss.backward()
            GenOptim.step()

            #######################
            #### Discremenator ####
            #######################
            # forward propagation
            out_dis = net.discriminator(inputs, out_gen)

            # compute loss
            dis_loss, dis_loss_dic = loss_dis(out_dis)

            # backward
            dis_loss.backward()
            DisOptim.step()

            # log
            gen_losses.append(gen_loss.cpu().detach().numpy())
            dis_losses.append(dis_loss.cpu().detach().numpy())

            if iter_count % config.train.log_every == 0:
                logger.train.info('Gen-loss / Dis-loss at iter %d : %.5f, %.5f'\
                    % (iter_count, gen_loss.item(), dis_loss.item()))
                logger.train.figure('train/gen', gen_loss_dic, iter_count)
                logger.train.figure('train/dis', dis_loss_dic, iter_count)
        # log
        logger.train.info('Gen-loss/Dis-loss epoch %d : %.5f, %.5f' \
                % (e, np.mean(gen_losses), np.mean(dis_losses)))

        # validate
        net.eval()
        criterias = []
        losses = []

        with torch.no_grad():
            for batch in data_loaders['val']:
                # training data to device
                inputs = {
                        'cvmcep': batch['cvmcep'].to(device),
                        'wav': batch['wav'].to(device),
                        'cvwav': batch['cvwav'].to(device)
                }

                #######################
                ###### Generator ######
                #######################
                # forward propagation
                out_gen = net.generator(inputs)

                # compute loss
                gen_loss, gen_loss_dic = loss_gen(out_gen, inputs)

                # loss
                criterias.append(gen_loss_dic['multi_res_stft'])
                losses.append(gen_loss.item())


        # log
        criteria = np.mean(criterias)
        logger.val.info('Validation loss at epoch %d: %.5f' % (e, np.mean(losses)))
        logger.val.info('Validation criteria: %.5f' % criteria)
        logger.val.figure('val/gen', gen_loss_dic, e)

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
                    'GenOptim': GenOptim,
                    'DisOptim': DisOptim,
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
    parser.add_argument('--conf_path', default=None, type=str,
                        help='Path to the config file')
    parser.add_argument('--model_dir', default=None, type=str,
                        help='Path to the directory where trained model will be saved.')
    parser.add_argument('--model_name', default=None, type=str,
                        help='Model name.')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Name log file will be saved')
    parser.add_argument('--resume', default=None, type=str,
                        help='Path to the checkpoint file.')
    args = parser.parse_args()

    # number pf speakers
    n_spk = 2

    # train
    train(args, n_spk=n_spk)
