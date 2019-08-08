from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import numpy as np

from preprocess_utils import write_to_hdf5_file


def filter_non_digits(root_dir):
    all_data_dir = os.path.join(root_dir, 'all_data')
    fnames = os.listdir(all_data_dir)
    fnames = [f for f in fnames if f.endswith('json')]

    out_dir = os.path.join(root_dir, 'digit_data')
    os.makedirs(out_dir)

    for fname in fnames:
        path = os.path.join(all_data_dir, fname)
        with open(path, 'r') as f:
            data = json.load(f)

        users = []
        num_samples = []
        user_data = {}

        for idx, user in enumerate(data['users']):
            xs = data['user_data'][user]['x']
            ys = data['user_data'][user]['y']

            new_xs = []
            new_ys = []

            for x, y in zip(xs, ys):
                if y >= 0 and y <= 9:
                    # digits
                    new_xs.append(x)
                    new_ys.append(y)

            users.append(user)
            num_samples.append(len(new_xs))
            user_data[user] = {'x': new_xs, 'y': new_ys}

            print('user: {}, #samples: {} -> {}'.format(
                user, data['num_samples'][idx], len(new_ys)))

        all_data = {}
        all_data['users'] = users
        all_data['num_samples'] = num_samples
        all_data['user_data'] = user_data

        out_path = os.path.join(out_dir, fname)
        print('writing to', out_path)
        with open(out_path, 'w') as out_f:
            json.dump(all_data, out_f)


def generate_hdf5_files(root_dir):

    def json_to_hdf5(root_dir, tag):
        json_dir = os.path.join(root_dir, tag)

        users = []
        xs = []
        ys = []

        for fname in os.listdir(json_dir):
            with open(os.path.join(json_dir, fname), 'r') as f:
                data = json.load(f)

            for user in data['users']:
                users.append(user)
                x = np.array(data['user_data'][user]['x'], dtype=np.float32)
                x = np.reshape(x, (x.shape[0], 28, 28))
                xs.append(x)
                y = np.array(data['user_data'][user]['y'], dtype=np.int32)
                ys.append(y)

        out_path = os.path.join(root_dir, 'femnist_digitsonly_%s.h5' % tag)
        print('writing %s data to %s' % (tag, out_path))
        write_to_hdf5_file(out_path, users, 'pixels', 'label', xs, ys)

    json_to_hdf5(root_dir, 'train')
    json_to_hdf5(root_dir, 'test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='femnist root path')
    parser.add_argument('--filter', action='store_true',
                        help='filter non-digit data')
    parser.add_argument('--hdf5', action='store_true',
                        help='generate hdf5 files')

    args = parser.parse_args()

    if args.filter:
        filter_non_digits(args.root_dir)
    elif args.hdf5:
        generate_hdf5_files(args.root_dir)
    else:
        raise ValueError('No action is given')


if __name__ == "__main__":
    main()
