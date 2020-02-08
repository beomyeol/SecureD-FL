from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

import datasets.femnist as femnist
import datasets.shakespeare as shakespeare
import datasets.vehicle as vehicle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='dataset name')
    parser.add_argument('--dir', required=True, help='dataset dir path')

    args = parser.parse_args()

    dataset_kwargs = {}
    name = args.name.lower()
    if name == 'femnist':
        dataset = femnist
        dataset_kwargs = {'only_digits': True}
    elif name == 'shakespeare':
        dataset = shakespeare
        dataset_kwargs = {'seq_length': 1}
    elif name == 'vehicle':
        dataset = vehicle
    else:
        raise ValueError('Unknown dataset: %s' % name)

    train_dataset = dataset.load_dataset(
        args.dir, train=True, dataset_download=True, **dataset_kwargs)

    test_dataset = dataset.load_dataset(
        args.dir, train=False, **dataset_kwargs)

    num_users = len(train_dataset.client_ids())
    client_train_datasets = [train_dataset.get_client_dataset(client_id)
                             for client_id in train_dataset.client_ids()]
    client_test_datasets = [test_dataset.get_client_dataset(client_id)
                            for client_id in test_dataset.client_ids()]
    num_train_list = np.array([len(client_dataset)
                      for client_dataset in client_train_datasets])
    num_test_list = np.array([len(client_dataset)
                     for client_dataset in client_test_datasets])

    print('# users:', num_users)
    print('# total train samples:', np.sum(num_train_list))
    print('# total test samples:', np.sum(num_test_list))
    print('min # train samples:', np.min(num_train_list))
    print('min # test samples:', np.min(num_test_list))

    num_total_samples_per_client = num_train_list + num_test_list
    print('train ratio:', np.mean(num_train_list/num_total_samples_per_client))
    print('test ratio:', np.mean(num_test_list/num_total_samples_per_client))

if __name__ == "__main__":
    main()
