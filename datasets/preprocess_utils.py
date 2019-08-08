from __future__ import absolute_import, division, print_function

import h5py
import numpy as np


def normalize(X):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)

    X = (X - mean) / stdev
    return X


def write_to_hdf5_file(path, unames, x_name, y_name, Xs, Ys):
    with h5py.File(path, 'w') as f:
        grp = f.create_group('examples')
        for uname, X, Y in zip(unames, Xs, Ys):
            user_grp = grp.create_group(uname)
            user_grp.create_dataset(x_name, data=X, compression='gzip')
            user_grp.create_dataset(y_name, data=Y, compression='gzip')
