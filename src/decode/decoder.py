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


IRLEN = 1024
INTERVALS = 10
SEED = 1
LP_CUTOFF = 20


class Decoder(object):
    def __init__(self, args, scaler, logger=None):
        # directory to save wav files
        self.save_dir = args.exp_dir
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
                    alpha=self.mcep_alpha,
                    rmcep=mcep
            )
            wav = np.clip(wav, -32768, 32767)
            wav_file = os.path.join(
                        wav_dir,
                        '%s_%s_%d.wav' % (inputs['src'][b], inputs['src'][b], i)
            )
            wavfile.write(wav_file, self.fs, wav.astype(np.int16))

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
                    alpha=self.mcep_alpha,
                    rmcep=mcep
            )
            wav = np.clip(wav, -32768, 32767)
            wav_file = os.path.join(
                        wav_dir,
                        '%s_%s_%d.wav' % (inputs['src'][b], inputs['trg'][b], i)
            )
            wavfile.write(wav_file, self.fs, wav.astype(np.int16))
