# Script for dataset

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train_dir, scaler_path, logger, pad_len = 2500,
            batch_len=80, device=None):
        self.device = device
        self.logger = logger
        self.batch_len = batch_len
        self.pad_len = pad_len
        self.ref_list = ['src_id', 'trg_id', 'flen', 'src_spk', 'trg_spk']

        # get scaler
        self._get_scaler(scaler_path)

        # load data
        self._load_data(train_dir)

        # split data
        if self.batch_len > 0:
            self._split_data(batch_len)

        # scaler
        self._post_prepro()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _split_data(self, batch_len):
        # split data into some mini-batches with length of batch_num
        # with shift lwngth == (batch_num / 2)
        shift_len = batch_len // 2

        data = []
        for d in self.data:
            # n_min-batch
            n_mini = d['flen'] // shift_len - 1

            # split into mini-batch
            for i in range(n_mini):
                start = i * shift_len
                end = start + batch_len

                data.append(
                    {
                        'uv': d['uv'][start:end],
                        'ap': d['ap'][start:end],
                        'lcf0': d['lcf0'][start:end],
                        'f0': d['f0'][start:end],
                        'cv_f0': d['cv_f0'][start:end],
                        'mcep': d['mcep'][start:end],
                        'codeap': d['codeap'][start:end],
                        'src_code': d['src_code'][start:end],
                        'trg_code': d['trg_code'][start:end],
                        'src_id': d['src_id'],
                        'trg_id': d['trg_id'],
                        'src_spk': d['src_spk'],
                        'trg_spk': d['trg_spk'],
                        'flen': batch_len
                    }
                )
        self.data = data

    def _get_scaler(self, scaler_path):
        self.scaler = joblib.load(os.path.join(scaler_path, 'scalers.pkl'))

    def _post_prepro(self):
        # scaler
        self._apply_scaler()

        # to torch.Tensor
        self._tensor()

    def _tensor(self):
        # process over all data
        for i in range(len(self.data)):
            for k in self.data[i].keys():
                if not k in self.ref_list:
                    self.data[i][k] = torch.Tensor(self.data[i][k])

    def _apply_scaler(self):
        # process over all data
        for i in range(len(self.data)):
            # normalize mcep, lcf0, codeap
            self.data[i]['mcep'] = self._transform('mcep', self.data[i]['mcep'])
            self.data[i]['lcf0'] = self._transform('lcf0', self.data[i]['lcf0'])
            self.data[i]['codeap'] = self._transform('codeap', self.data[i]['codeap'])

    def _transform(self, k, x):
        m = self.scaler[k].mean_
        s = self.scaler[k].scale_
        return (x - m) / s

    def _low_pass_filter(self, x, fs=5, cutoff=20, padding=True):
        """FUNCTION TO APPLY LOW PASS FILTER

        Args:
            x (ndarray): Waveform sequence
            fs (int): Sampling frequency
            cutoff (float): Cutoff frequency of low pass filter

        Return:
            (ndarray): Low pass filtered waveform sequence
        """
        x = x[:, 0]

        nyquist = fs // 2
        norm_cutoff = cutoff / nyquist

        # low cut filter
        numtaps = 255
        fil = firwin(numtaps, norm_cutoff)
        x_pad = np.pad(x, (numtaps, numtaps), 'edge')
        lpf_x = lfilter(fil, 1, x_pad)
        lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

        return lpf_x[:, np.newaxis]

    def _convert_f0(self, f0, src, trg):
        f0 = f0[:, 0]
        nonzero_indicies = f0 > 0
        cvf0 = np.zeros(len(f0))
        cvf0[nonzero_indicies] = \
                (self.scaler[trg]['f0'].scale_ / self.scaler[src]['f0'].scale_) \
                * (f0[nonzero_indicies] - self.scaler[src]['f0'].mean_) \
                + self.scaler[trg]['f0'].mean_
        return cvf0[:, np.newaxis]

    def _get_spk_code(self, spk_names):
        ret = {}
        spk_mat = torch.Tensor(np.eye(len(spk_names)))
        for i, spk in enumerate(spk_names):
            ret[spk + '_id'] = i
            ret[spk + '_code'] = spk_mat[i]
        return ret

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

                for trg in trg_spks:
                    flen = f['mcep'].shape[0]
                    self.data.append(
                        {
                            'uv': self._read_feature(f, 'uv'),
                            'ap': self._read_feature(f, 'ap'),
                            'f0': self._read_feature(f, 'f0'),
                            'cv_f0': self._convert_f0(self._read_feature(f, 'f0'), spk, trg),
                            'lcf0': self._read_feature(f, 'lcf0'),
                            'mcep': self._read_feature(f, 'mcep'),
                            'codeap': self._read_feature(f, 'codeap'),
                            'src_code': spk_dict[spk + '_code'].unsqueeze(0).repeat(flen, 1),
                            'trg_code': spk_dict[trg + '_code'].unsqueeze(0).repeat(flen, 1),
                            'src_id': spk_dict[spk + '_id'],
                            'trg_id': spk_dict[trg + '_id'],
                            'flen': f['mcep'].shape[0],
                            'src_spk': spk,
                            'trg_spk': trg
                        }
                    )

    def _read_feature(self, data, key):
        d = data[key][:]
        if len(d.shape) == 1:
            return d[:, np.newaxis]
        else:
            return d
