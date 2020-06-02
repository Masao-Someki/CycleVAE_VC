
# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import fnmatch
import os
import sys
import threading
import librosa

import h5py
import numpy as np


def read_hdf5(hdf5_name, hdf5_path):
    """FUNCTION TO READ HDF5 DATASET
    Args:
        hdf5_name (str): filename of hdf5 file
        hdf5_path (str): dataset name in hdf5 file
    Return:
        dataset values
    """
    if not os.path.exists(hdf5_name):
        print("ERROR: There is no such a hdf5 file. (%s)" % hdf5_name)
        print("Please check the hdf5 file path.")
        sys.exit(-1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        print("ERROR: There is no such a data in hdf5 file. (%s)" % hdf5_path)
        print("Please check the data path in hdf5 file.")
        sys.exit(-1)

    hdf5_data = hdf5_file[hdf5_path].value
    hdf5_file.close()

    return hdf5_data



def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """FUNCTION TO WRITE DATASET TO HDF5
    Args :
        hdf5_name (str): hdf5 dataset filename
        hdf5_path (str): dataset path in hdf5
        write_data (ndarray): data to write
        is_overwrite (bool): flag to decide whether to overwrite dataset
    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                print("Warning: data in hdf5 file already exists. recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                print("ERROR: there is already dataset.")
                print("if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()

    return 1

def remove_breath(audio):
    mask = np.zeros(audio.shape)
    edges = librosa.effects.split(
            audio, top_db=20, frame_length=16, hop_length=8)
    for idx in range(len(edges)):
        start_idx, end_idx = edges[idx][0], edges[idx][1]
        if start_idx < len(audio):
            mask[start_idx:end_idx] = 1
    return mask * audio


