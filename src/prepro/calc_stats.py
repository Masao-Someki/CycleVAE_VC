#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate acoustic feature statistics

"""

import argparse
import os
import sys
import glob
import joblib
import logging

from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np

class Scaler(object):
    def __init__(self):
        self.ss = StandardScaler()

    def partial_fit(self, data):
        self.ss.partial_fit(data)

    def fit(self, file_lists, ext='wmcep'):
        for h5f in file_lists:
            with h5py.File(h5f, 'r') as fp:
                data = fp[ext][:]
                if 'f0' == ext:
                    nonzero_indicies = data > 0
                    data = data[nonzero_indicies]
                if len(data.shape) == 1:
                    data = data[:, np.newaxis]
                self.partial_fit(data)

def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--speaker', default=None, type=str,
                        help='Name of the speaker')
    parser.add_argument('--stats_dir', default=None, type=str,
                        help='Path to the directory to save the stats file.')
    parser.add_argument('--hdf5_dir', type=str,
                        help='Statistics directory of the speaker')
    args = parser.parse_args()

    # open h5 files
    all_spks = [os.path.basename(f) for f in glob.glob(os.path.join(args.hdf5_dir, '*'))]
    h5_files = []
    for spk in all_spks:
        for file_name in glob.glob(os.path.join(args.hdf5_dir, spk, '*.h5')):
            h5_files.append(file_name)

    # Speaker Independent-scaler extraction
    feats = ['mcep', 'lcf0', 'codeap']
    scalers = {}
    for ext in feats:
        s = Scaler()
        s.fit(h5_files, ext=ext)
        logging.info("# of samples for {}: {}".format(ext, s.ss.n_samples_seen_))
        scalers[ext] = s.ss

    # Speaker Dependent-scaler extraction
    for spk in all_spks:
        spk_files = glob.glob(os.path.join(args.hdf5_dir, spk, '*.h5'))
        scalers[spk] = {}

        for feat in ['lcf0', 'f0']:
            s = Scaler()
            s.fit(spk_files, ext=feat)
            logging.info("# of samples for {} : {} {} samples".format(
                spk, feat, s.ss.n_samples_seen_))
            scalers[spk][feat] = s.ss

    pkl_path = os.path.join(args.stats_dir, 'scalers.pkl')
    joblib.dump(scalers, pkl_path)
    logging.info("Save scaler: {}".format(pkl_path))


if __name__ == '__main__':
    main()
