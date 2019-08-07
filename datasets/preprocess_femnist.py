from __future__ import absolute_import, division, print_function

import argparse
import json
import os


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
            user_data = data['user_data'][user]

            new_xs = []
            new_ys = []

            for x, y in zip(user_data['x'], user_data['y']):
                if y >= 0 and y <= 9:
                    # digits
                    new_xs.append(x)
                    new_ys.append(y)

            if len(new_xs) > 0:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='femnist root path')

    args = parser.parse_args()

    filter_non_digits(args.root_dir)


if __name__ == "__main__":
    main()
