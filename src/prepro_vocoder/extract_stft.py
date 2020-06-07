# compute stft and mag.
import os
import argparse
import h5py
import glob
import multiprocessing as mp

import numpy as np
import librosa

from writer import Logger


def prepro_for_vocoder(args):
    # set logger
    logger = Logger(args.log_name, 'preprocess').preprocess

    # get file list
    h5_list = glob.glob(os.path.join(args.hdf5dir, '*.h5'))

    # compute stft
    def feature_extract(h5_list):
        logger.info('Process %d samples' % len(h5_list))

        for h5_file in h5_list:
            # log
            logger.info('processing %s' % h5_file)

            # build wav path
            wav_file = os.path.join(
                            args.wav_dir,
                            os.path.basename(h5_file).replace('h5', 'wav')
                            )

            # load wav file
            x, _ = librosa.load(wav_file, args.fs)

            # save dataset
            with h5py.File(h5_file, 'a') as f:
                # check if key already exists
                if 'wav' in f.keys():
                    del f['wav']

                # save
                f.create_dataset('wav', data=x)

    # run with multiprocessing
    # divie list
    file_lists = np.array_split(h5_list, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multiprocessing
    processes = []
    for f in file_lists:
        p = mp.Process(target=feature_extract, args=(f,))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()

    parser.add_argument('--hdf5dir', default=None, type=str,
                        help='Path to the hdf5 data dir')
    parser.add_argument('--wav_dir', default=None, type=str,
                        help='Path to the wav files')
    parser.add_argument('--log_name', default=None, type=str,
                        help='Log name.')
    parser.add_argument('--fs', default=None, type=int,
                        help='Sampling rate')
    parser.add_argument('--n_jobs', default=1, type=int,
                        help='Number of cpus to use.')

    args = parser.parse_args()

    # preprocess wav data for vocoder training.
    prepro_for_vocoder(args)
