# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)
import os
import glob
import h5py
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from net import Net
from dataset import Dataset
from net import Optimizers
from writer import Logger
from utils import get_config
from decode import Decoder

from speech import Synthesizer

np.random.seed(4)
torch.manual_seed(4)


class SimpleDataset(Dataset):
    def _load_data(self, train_dir):
        self.data = []
        self.all_spks = [os.path.basename(i) for i in glob.glob(os.path.join(train_dir, '*'))]
        spk_dict = self._get_spk_code(self.all_spks)

        # load all data
        for spk in self.all_spks:
            # load data
            for file_path in glob.glob(os.path.join(train_dir, spk, '*.h5')):
                f = h5py.File(file_path, 'r')
                trg_spks = [s for s in self.all_spks if not s == spk]


                flen = f['mcep'].shape[0]
                self.data.append(
                    {
                        'uv': self._read_feature(f, 'uv'),
                        'f0': self._read_feature(f, 'f0'),
                        'lcf0': self._read_feature(f, 'lcf0'),
                        'codeap': self._read_feature(f, 'codeap'),
                        'mcep': self._read_feature(f, 'mcep'),
                        'src_code': spk_dict[spk + '_code'].unsqueeze(0).repeat(flen, 1),
                        'trg_code': spk_dict[spk + '_code'].unsqueeze(0).repeat(flen, 1),
                        'file_path': file_path
                    }
                )
            self.ref_list = ['file_path']

class SimpleDecoder(Decoder):
    def __init__(self, args, scaler, logger=None):
        # directory to save wav files
        self.fs = args.fs
        self.shiftms = args.shiftms
        self.fftl = args.fftl

        # mcep_alpha
        if args.fs == 16000:
            self.mcep_alpha = 0.41
        elif args.fs == 22050:
            self.mcep_alpha = 0.455
        elif args.fs == 24000:
            self.mcep_alpha = 0.466
        elif args.fs == 44100:
            self.mcep_alpha = 0.544
        elif args.fs == 48000:
            self.mcep_alpha = 0.554
        else:
            raise ValueError('sampling rate should be one of  \
                16000, 22050, 24000, 44100, 48000')

        # scaler
        self.scaler = scaler

        # synthesizer
        self.synthesizer = Synthesizer(fs=args.fs, fftl=args.fftl, shiftms=args.shiftms)

        # logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def decode(self, inputs, output):
        # mcep
        mcep = inputs['mcep'][0].cpu().detach().numpy()
        mcep = self._inverse_transform('mcep', mcep).astype(np.float64)

        # process src-src wav
        cvmcep = output['reconst_half'][0][0].cpu().detach().numpy()
        cvmcep = self._inverse_transform('mcep', cvmcep).astype(np.float64)

        # codeap
        codeap = inputs['codeap'][0].cpu().detach().numpy()
        codeap = self._inverse_transform('codeap', codeap).astype(np.float64)

        # synthesize
        wav = self.synthesizer.synthesis(
                inputs['f0'][0].squeeze(1).cpu().detach().numpy().astype(np.float64),
                cvmcep,
                codeap,
                alpha=self.mcep_alpha,
                rmcep=mcep
        )

        wav = np.clip(wav, -32768, 32767)
        wav = wav / 32767

        return wav


def decode(aegs, n_spk):
    # load config
    config = get_config(args.conf_path)

    # logger
    logger = Logger(args.log_name, 'convert', 'dataset')

    # training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.convert.info('device: %s' % device)

    # trainind settings and model
    net = Net(config.model, n_spk, 1, device)
    net.to(device)

    # resume
    dic = torch.load(args.checkpoint)
    net.load_state_dict(dic['model'])
    criteria_before = dic['criteria']
    iter_count = dic['iter_count']
    logger.convert.info(net)
    logger.convert.info('Criteria before: %f' % criteria_before)

    # dataset
    datasets = {'convert': SimpleDataset(args.data_dir, args.stats_dir,
                                 logger.dataset, pad_len=2800,
                                 batch_len=-1, device=device)
                }

    data_loaders = {'convert': DataLoader(datasets['convert'],
                                        batch_size=1,
                                        shuffle=False)
                    }

    # decoder
    decoder = SimpleDecoder(args, datasets['convert'].scaler, logger=logger.convert)

    # logging about training data
    logger.dataset.info('number of samples: %d' % len(datasets['convert']))

    # decode
    logger.convert.info('start decoding!')

    # log parameter
    count = 0

    # convert mcep to cvmcep
    with torch.no_grad():
        for i, batch in enumerate(data_loaders['convert']):
            # log
            count += 1
            logger.convert.info('processing %s (%d/%d)' % \
                (batch['file_path'][0], count, len(datasets['convert'])))

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
                        'mcep': batch['mcep'],
                        'f0': batch['f0'],
                        'codeap': batch['codeap']
                }

            # forward propagation with target-pos output
            outputs = net(inputs)

            # decode
            wav = decoder.decode(inputs, outputs)

            # save
            with h5py.File(batch['file_path'][0], 'a') as f:

                # check if the key already exists
                if 'cvmcep' in f.keys():
                    # delete value
                    del f['cvmcep']

                if 'cvwav' in f.keys():
                    del f['cvwav']

                # save converted mcep
                f.create_dataset('cvmcep',
                        data=outputs['reconst_half'][0].squeeze(0)\
                        .cpu().detach().numpy()
                )
                f.create_dataset('cvwav', data=wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str,
                        help='Path to the training data.')
    parser.add_argument('--stats_dir', default=None, type=str,
                        help='Path to the stats files.')
    parser.add_argument('--conf_path', default=None, type=str,
                        help='Path to the config file')
    parser.add_argument("--fs", default=None,
                        type=int, help="Sample rate.")
    parser.add_argument("--shiftms", default=None,
                        type=int, help="Shift ms.")
    parser.add_argument("--mcep_dim", default=None,
                        type=int, help="Dimension of mel cepstrum")
    parser.add_argument("--fftl", default=None,
                        type=int, help="FFT length")
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='Path to the directory where trained model will be saved.')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Name log file will be saved')

    args = parser.parse_args()

    n_spk = 2

    # decode
    decode(args, n_spk=n_spk)
