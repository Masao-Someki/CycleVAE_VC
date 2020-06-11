# Script to compute dataset

# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)

import os
import glob
import joblib

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import firwin, lfilter
import torch

from .dataset import Dataset


class VocoderDataset(Dataset):
    def __init__(self, train_dir, scaler_path, logger, hop_size, batch_len=80, device=None):
            self.hop_size = hop_size
            self.device = device
            self.logger = logger
            self.batch_len = batch_len

            # get scaler
            self._get_scaler(scaler_path)

            # load data
            self._load_data(train_dir)

            # split data
            if self.batch_len > 0:
                self._split_data(batch_len)

            # scaler
            self._post_prepro()

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

                self.data.append(
                    {
                        'cvmcep': self._read_feature(f, 'cvmcep'),
                        'wav': self._read_feature(f, 'wav'),
                        'cvwav': self._read_feature(f, 'cvwav')
                    }
                )

    def _apply_scaler(self):
        # process over all data
        for i in range(len(self.data)):
            # normalize mcep, lcf0, codeap
            self.data[i]['cvmcep'] = self._transform('mcep', self.data[i]['cvmcep'])

    def _tensor(self):
        # process over all data
        for i in range(len(self.data)):
            for k in self.data[i].keys():
                self.data[i][k] = torch.Tensor(self.data[i][k]).transpose(0, 1)

    def _split_data(self, batch_len):
        # split data into some mini-batches with length of batch_num
        # with shift lwngth == (batch_num / 2)
        shift_len = batch_len // 2

        data = []
        for d in self.data:
            # flen
            flen = len(d['cvmcep'])

            # n_min-batch
            n_mini = flen // shift_len - 1

            # split into mini-batch
            for i in range(n_mini):
                start = i * shift_len
                end = start + batch_len

                cvmcep = d['cvmcep'][start:end]
                wav = d['wav'][start * self.hop_size : end * self.hop_size]
                cvwav = d['cvwav'][start * self.hop_size : end * self.hop_size]

                if not wav.shape == cvwav.shape:
                    continue

                data.append(
                    {
                        'cvmcep': d['cvmcep'][start:end],
                        'wav': d['wav'][start * self.hop_size : end * self.hop_size],
                        'cvwav': d['cvwav'][start * self.hop_size : end * self.hop_size]
                    }
                )

        self.data = data
