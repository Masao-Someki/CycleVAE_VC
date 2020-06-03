#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2019 Patrick Lumban Tobing (Nagoya University)
# based on PyTorch implementation for WaveNet vocoder by Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Modified by Masao Someki

from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import sys
import glob

import logging
import numpy as np
from numpy.matlib import repmat
from scipy.interpolate import interp1d
from scipy.signal import firwin
from scipy.signal import lfilter
from scipy.io import wavfile

import librosa

from prepro.utils import write_hdf5, read_hdf5

from multiprocessing import Array

import pysptk as ps
import pyworld as pw

IRLEN = 1024
LOWPASS_CUTOFF = 20
HIGHPASS_CUTOFF = 70
OVERWRITE = True


def low_cut_filter(x, fs, cutoff=HIGHPASS_CUTOFF):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def analyze(wav, fs=16000, minf0=70, maxf0=700, fperiod=5, fftl=1024, f0=None, time_axis=None):
    #f0_flr = pw.get_cheaptrick_f0_floor(fs, fftl)
    #logging.info(f0_flr)
    #fft_size = pw.get_cheaptrick_fft_size(fs, f0_flr)
    #logging.info(fft_size)
    #f0_flr = pw.get_cheaptrick_f0_floor(fs, fft_size)
    #logging.info(f0_flr)
    if f0 is None or time_axis is None:
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=60.0, frame_period=fperiod)
        f0 = pw.stonemask(wav, _f0, time_axis, fs)
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)

    return time_axis, f0, sp, ap


def analyze_range(wav, fs=16000, minf0=70, maxf0=700, fperiod=5, fftl=1024, f0=None, time_axis=None):
    if f0 is None or time_axis is None:
        _f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, f0_ceil=maxf0, frame_period=fperiod)
        f0 = pw.stonemask(wav, _f0, time_axis, fs)
        #f0, time_axis = pw.harvest(wav, fs, f0_floor=minf0, f0_ceil=maxf0, frame_period=fperiod)
    sp = pw.cheaptrick(wav, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(wav, f0, time_axis, fs, fft_size=fftl)

    return time_axis, f0, sp, ap


def read_wav(wav_file, cutoff=HIGHPASS_CUTOFF):
    fs, x = wavfile.read(wav_file)
    x = np.array(x, dtype=np.float64)
    if cutoff != 0:
        x = low_cut_filter(x, fs, cutoff)

    return fs, x


def convert_f0(f0, f0_mean_src, f0_std_src, f0_mean_trg, f0_std_trg):
    nonzero_indices = f0 > 0
    cvf0 = np.zeros(len(f0))
    cvf0[nonzero_indices] = np.exp((f0_std_trg/f0_std_src)*(np.log(f0[nonzero_indices])-f0_mean_src)+f0_mean_trg)

    return cvf0


def convert_linf0(f0, f0_mean_src, f0_std_src, f0_mean_trg, f0_std_trg):
    nonzero_indices = f0 > 0
    cvf0 = np.zeros(len(f0))
    cvf0[nonzero_indices] = (f0_std_trg/f0_std_src)*(f0[nonzero_indices]-f0_mean_src)+f0_mean_trg

    return cvf0

def mod_pow(cvmcep, mcep, alpha=0.41, irlen=IRLEN):
    cv_e = ps.mc2e(cvmcep, alpha=alpha, irlen=irlen)
    r_e = ps.mc2e(mcep, alpha=alpha, irlen=irlen)
    dpow = np.log(r_e/cv_e) / 2
    mod_cvmcep = np.copy(cvmcep)
    mod_cvmcep[:,0] += dpow

    return mod_cvmcep


def extfrm(data, npow, power_threshold=-20):
    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata, valid_index


def spc2npow(spectrogram):
    npow = np.apply_along_axis(spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow/meanpow)

    return npow


def spvec2pow(specvec):
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def low_pass_filter(x, fs, cutoff=LOWPASS_CUTOFF, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]

    return lpf_x


def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def calc_jnt_sdmat(mcep, coeff):
    assert(len(coeff) == 3)

    return np.concatenate([mcep,np.insert(mcep[:-1,:]*coeff[0], 0, 0.0, axis=0) + mcep*coeff[1] + np.append(mcep[1:,:]*coeff[2], np.zeros((1,mcep.shape[1])), axis=0)], axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="making feature file argsurations.")

    parser.add_argument("--log_dir", required=True,
        type=str, help="directory to save the log")
    parser.add_argument(
        "--wav_dir", default=None,
        help="directory of input wavfile")
    parser.add_argument(
        "--hdf5dir", default=None,
        help="directory to save hdf5")
    parser.add_argument(
        "--conf_path", default=None,
        type=str, help="Path to the config file")
    parser.add_argument(
        "--fs", default=None,
        type=int, help="Sample rate.")
    parser.add_argument(
        "--shiftms", default=None,
        type=int, help="Shift ms.")
    parser.add_argument(
        "--mcep_dim", default=None,
        type=int, help="Dimension of mel cepstrum")
    parser.add_argument(
        "--fftl", default=None,
        type=int, help="FFT length")
    parser.add_argument(
        "--n_jobs", default=1,
        type=int, help="number of parallel jobs")
    args = parser.parse_args()

    # set log level
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S',
                        filename=args.log_dir + "/feature_extract.log")
    logging.getLogger().addHandler(logging.StreamHandler())

    # mcep_alpha
    if args.fs == 16000:
        mcep_alpha = 0.41
    elif args.fs == 22050:
        mcep_alpha = 0.455
    elif args.fs == 24000:
        mcep_alpha = 0.466
    elif args.fs == 44100:
        mcep_alpha = 0.544
    elif args.fs == 48000:
        mcep_alpha = 0.554
    else:
        raise ValueError('sampling rate should be one of  \
            16000, 22050, 24000, 44100, 48000')

    # read list
    file_list = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))

    # load config
    with open(args.conf_path, 'r', encoding='utf-8') as f:
        t = f.readlines()

    minf0 = int(t[0])
    maxf0 = int(t[1])
    thpow = float(t[2])

    def feature_extract(wav_list, arr):
        n_sample = 0
        n_frame = 0
        max_frame = 0
        count = 1
        coeff = np.array([-0.5,0.5,0.0])
        for wav_name in wav_list:
            # load wavfile and apply low cut filter
            fs, x = read_wav(wav_name, cutoff=70)
            n_sample += x.shape[0]
            logging.info(wav_name+" "+str(x.shape[0])+" "+str(n_sample)+" "+str(count))

            # check sampling frequency
            if not fs == args.fs:
                logging.debug("ERROR: sampling frequency is not matched.")
                sys.exit(1)

            hdf5name = args.hdf5dir + "/" + os.path.basename(wav_name).replace(".wav", ".h5")

            # extimate f0 and ap
            time_axis, f0, spc, ap = analyze_range(x, fs=args.fs, minf0=minf0,
                maxf0=maxf0, fperiod=args.shiftms, fftl=args.fftl)
            write_hdf5(hdf5name, '/ap', ap)
            write_hdf5(hdf5name, "/f0", f0)

            # convert to continuous f0 and low-pass filter
            uv, cont_f0 = convert_continuos_f0(np.array(f0))
            cont_f0_lpf = low_pass_filter(cont_f0,
                int(1.0 / (args.shiftms * 0.001)), cutoff=20)

            cont_f0_lpf = np.expand_dims(cont_f0_lpf, axis=-1)
            uv = np.expand_dims(uv, axis=-1)

            write_hdf5(hdf5name, "/lcf0", np.log(cont_f0_lpf))
            write_hdf5(hdf5name, "/uv", uv)

            # extimate codeap
            codeap = pw.code_aperiodicity(ap, args.fs)
            if codeap.ndim == 1:
                # when fs == 16000
                codeap = np.expand_dims(codeap, axis=-1)
            write_hdf5(hdf5name, "/codeap", codeap)

            # mcep
            mcep = ps.sp2mc(spc, args.mcep_dim, mcep_alpha)
            write_hdf5(hdf5name, "/mcep", mcep)

    # divie list
    file_lists = np.array_split(file_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    processes = []
    arr = mp.Array('d', 3)
    for f in file_lists:
        p = mp.Process(target=feature_extract, args=(f,arr))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
