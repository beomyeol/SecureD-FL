from __future__ import absolute_import, division, print_function

import argparse
from collections import namedtuple
import tempfile
import numpy as np
import h5py
import os
from torchvision.datasets import CIFAR10
from torchvision import transforms

DatasetAttr = namedtuple(
    'DatasetAttr', ['cls', 'transform', 'xname', 'yname', 'xtype', 'ytype'])


class ImageToNumpy(object):

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        return self.to_tensor(img).numpy()


_DATASET_ATTR = {
    "cifar10": DatasetAttr(CIFAR10, transform=ImageToNumpy(),
                           xname='pixels', xtype=np.float32,
                           yname='label', ytype=np.int32)
}


def split_dataset(dataset, seed, num_clients):
    indexes = list(np.arange(len(dataset)))
    np.random.seed(seed)
    np.random.shuffle(indexes)

    dataset_splits = []

    for i, split in enumerate(np.array_split(indexes, num_clients)):
        xs, ys = [], []
        for idx in split:
            x, y = dataset[idx]
            xs.append(x)
            ys.append(y)

        dataset_splits.append(('f_%d' % i, xs, ys, split))

    return dataset_splits


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


def generate_dataset(dataset, dataset_root, seed, num_clients, out_dir):
    dataset_attr = _DATASET_ATTR[dataset]

    if not dataset_root:
        temp_dir = tempfile.TemporaryDirectory()
        dataset_root = temp_dir.name
    else:
        temp_dir = None

    train_dataset = dataset_attr.cls(
        dataset_root, train=True, download=True,
        transform=dataset_attr.transform)
    test_dataset = dataset_attr.cls(
        dataset_root, train=False, download=True,
        transform=dataset_attr.transform)

    train_dataset_splits = split_dataset(train_dataset, seed, num_clients)
    test_dataset_splits = split_dataset(test_dataset, seed, num_clients)

    os.makedirs(out_dir, exist_ok=True)

    write_dataset_to_hdfs_file(
        os.path.join(out_dir, '%s_train.h5' % dataset),
        train_dataset_splits,
        seed,
        xname=dataset_attr.xname, yname=dataset_attr.yname,
        xtype=dataset_attr.xtype, ytype=dataset_attr.ytype)

    write_dataset_to_hdfs_file(
        os.path.join(out_dir, '%s_test.h5' % dataset),
        test_dataset_splits,
        seed,
        xname=dataset_attr.xname, yname=dataset_attr.yname,
        xtype=dataset_attr.xtype, ytype=dataset_attr.ytype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='dataset name to generate.')
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
