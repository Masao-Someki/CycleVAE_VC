# Copyright 2020 Masao Someki
#  MIT License (https://opensource.org/licenses/MIT)
import os
import glob
import h5py
import logging

import librosa
import numpy as np
from scipy.io import wavfile

from speech import Synthesizer

FS = 24000
#FS = 22050
N_GPUS = 1
SHIFT_MS = 5
#MCEP_ALPHA = 0.41000000000000003
#MCEP_ALPHA = 0.455
MCEP_ALPHA = 0.466
FFTL = 1024
IRLEN = 1024
INTERVALS = 10
SEED = 1
LP_CUTOFF = 20


class Decoder(object):
    def __init__(self, save_dir, scaler, logger=None):
        # directory to save wav files
        self.save_dir = save_dir

        # scaler
        self.scaler = scaler

        # synthesizer
        self.synthesizer = Synthesizer(fs=FS, fftl=FFTL, shiftms=SHIFT_MS)

        # logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def _inverse_transform(self, key, x):
        m = self.scaler[key].mean_
        s = self.scaler[key].scale_
        return x * s + m

    def decode(self, inputs, output, iter_count, i):
        # directory
        wav_dir = os.path.join(self.save_dir, str(iter_count))

        if not os.path.exists(wav_dir):
            os.mkdir(wav_dir)

        # process over all data
        for b in range(len(output['reconst_half'][0])):
            # flen
            flen = inputs['flen'][b]

            # mcep
            mcep = inputs['mcep'][b][:flen].cpu().detach().numpy()
            mcep = self._inverse_transform('mcep', mcep).astype(np.float64)

            # process src-src wav
            cvmcep = output['reconst_half'][0][b][:flen].cpu().detach().numpy()
            cvmcep = self._inverse_transform('mcep', cvmcep).astype(np.float64)

            # codeap
            codeap = inputs['codeap'][b][:flen].cpu().detach().numpy().astype(np.float64)
            codeap = self._inverse_transform('codeap', codeap)

            # synthesize
            wav = self.synthesizer.synthesis(
                    inputs['f0'][b][:flen].squeeze(1).cpu().detach().numpy().astype(np.float64),
                    cvmcep,
                    codeap,
                    alpha=MCEP_ALPHA,
                    rmcep=mcep
            )
            wav = np.clip(wav, -32768, 32767)
            wav_file = os.path.join(
                        wav_dir,
                        '%s_%s_%d.wav' % (inputs['src'][b], inputs['src'][b], i)
            )
            wavfile.write(wav_file, FS, wav.astype(np.int16))

            # process src-trg wav
            cvmcep = output['trg_reconst'][b][:flen].cpu().detach().numpy()
            cvmcep = self._inverse_transform('mcep', cvmcep).astype(np.float64)

            # convert f0
            cvf0 = inputs['cv_f0'][b][:flen].squeeze(1).cpu().detach().numpy().astype(np.float64)

            # synthesize
            wav = self.synthesizer.synthesis(
                    cvf0,
                    cvmcep,
                    codeap,
                    alpha=MCEP_ALPHA,
                    rmcep=mcep
            )
            wav = np.clip(wav, -32768, 32767)
            wav_file = os.path.join(
                        wav_dir,
                        '%s_%s_%d.wav' % (inputs['src'][b], inputs['trg'][b], i)
            )
            wavfile.write(wav_file, FS, wav.astype(np.int16))
