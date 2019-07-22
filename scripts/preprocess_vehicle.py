from __future__ import absolute_import, division, print_function

import argparse
import h5py
import scipy.io
import numpy as np
import os


DEFAULTS = {
    'test_ratio': 0.25,
}


def loadmat(path):
    mat = scipy.io.loadmat(path)
    return mat['X'], mat['Y']


def normalize(X):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)

    X = (X - mean) / stdev
    return X


def write_to_file(path, unames, Xs, Ys):
    with h5py.File(path, 'w') as f:
        grp = f.create_group('examples')
        for uname, X, Y in zip(unames, Xs, Ys):
            user_grp = grp.create_group(uname)
            user_grp['values'] = X
            user_grp['labels'] = Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'INPUT', nargs=1, help='matlab input file path')
    parser.add_argument(
        '--outdir', required=True, help='output dir')
    parser.add_argument(
        '--normalize', action='store_true', help='normalize input data')
    parser.add_argument(
        '--test_ratio', type=float,
        default=DEFAULTS['test_ratio'],
        help='ratio of test data (default={})'.format(DEFAULTS['test_ratio']))

    args = parser.parse_args()

    X, Y = loadmat(args.INPUT[0])
    assert len(X) == len(Y), '#users must be the same'
    print('#users: %d' % len(X))

    Xs = [x[0] for x in X]
    Ys = [y[0] for y in Y]

    if args.normalize:
        print('Normalizing data...')
        Xs = [normalize(X) for X in Xs]

    os.makedirs(args.outdir, exist_ok=True)

    train_out = os.path.join(args.outdir, 'vehicle_train.h5')
    test_out = os.path.join(args.outdir, 'vehicle_test.h5')

    unames = ['f{0:02d}'.format(i) for i in range(len(X))]
    train_Xs = []
    train_Ys = []
    test_Xs = []
    test_Ys = []

    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        num_train_samples = int((1-args.test_ratio) * len(X))
        train_X = X[:num_train_samples]
        train_Y = Y[:num_train_samples]
        test_X = X[num_train_samples:]
        test_Y = Y[num_train_samples:]

        print('uname: %s, #train_samples: %d, #test_samples: %d' %
              (unames[i], len(train_X), len(test_X)))

        train_Xs.append(train_X)
        train_Ys.append(train_Y)
        test_Xs.append(test_X)
        test_Ys.append(test_Y)

    write_to_file(train_out, unames, train_Xs, train_Ys)
    write_to_file(test_out, unames, test_Xs, test_Ys)


if __name__ == "__main__":
    main()
