from __future__ import absolute_import, division, print_function

import argparse
from collections import namedtuple
import tempfile
import numpy as np
import h5py
import os

import dataset_factory
from preprocess_utils import write_to_hdf5_file


def write_dataset_to_hdfs_file(path, dataset_splits, seed,
                               xname, yname, xtype, ytype):
    print('Writing to {}'.format(os.path.abspath(path)))
    with h5py.File(path, 'w') as f:
        f['seed'] = seed
        grp = f.create_group('examples')
        for uname, xs, ys, indexes in dataset_splits:
            user_grp = grp.create_group(uname)
            user_grp.create_dataset(
                xname, data=np.array(xs, dtype=xtype), compression='gzip')
            user_grp.create_dataset(
                yname, data=np.array(ys, dtype=ytype), compression='gzip')
            user_grp.create_dataset(
                'indexes', data=np.array(indexes), compression='gzip')


def generate_dataset(dataset_name, dataset_root, seed, num_clients, out_dir):
    if not dataset_root:
        temp_dir = tempfile.TemporaryDirectory()
        dataset_root = temp_dir.name
    else:
        temp_dir = None

    os.makedirs(out_dir, exist_ok=True)

    def write_dataset(train):
        dataset = dataset_factory.create_dataset(
            dataset_name,
            root_dir=dataset_root,
            train=train,
            download=True,
            seed=seed,
            num_clients=num_clients)

        unames, Xs, Ys = [], [], []
        for client_id in dataset.client_ids():
            client_dataset = dataset.get_client_dataset(client_id)
            unames.append(client_id)
            Xs.append(client_dataset.X)
            Ys.append(client_dataset.Y)

        fname = '{}_train.h5' if train else '{}_test.h5'
        path = os.path.join(out_dir, fname.format(dataset_name))
        write_to_hdf5_file(path, unames, 'pixels', 'label', Xs, Ys)
        with h5py.File(path, 'r+') as f:
            f['seed'] = seed

    write_dataset(train=True)
    write_dataset(train=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', help='dataset name to generate.')
    parser.add_argument('--dataset_root', default='',
                        help='root dir where the original dataset is stored.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed.')
    parser.add_argument('--num_clients', type=int, required=True,
                        help='num clients to generate.')
    parser.add_argument('--out_dir', default='.',
                        help='output dir to store the generated dataset.')

    args = parser.parse_args()

    generate_dataset(**vars(args))


if __name__ == "__main__":
    main()
