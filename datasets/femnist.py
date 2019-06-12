from __future__ import absolute_import, division, print_function

import h5py
import os.path
import torch.utils.data
import random

from datasets.utils import download_and_extract_archive


class FEMNISTClientDataset(torch.utils.data.Dataset):

    def __init__(self, client_h5, transform=None, target_transform=None):
        self._images = client_h5['pixels']
        self._labels = client_h5['label']
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index, :]
        label = self._labels[index].item()

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label


class FEMNISTDataset(object):

    _EXAMPLE_GROUP = 'examples'
    _BASE_URL = 'https://storage.googleapis.com/tff-datasets-public/'

    def __init__(self, root, train=True, download=False, transform=None,
                 target_transform=None, only_digits=False):
        self._root = root
        self._fileprefix = 'fed_emnist'
        if only_digits:
            self._fileprefix += '_digitsonly'

        if download:
            self.download()

        if train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self._h5_file = h5py.File(data_file, 'r')
        self._transform = transform
        self._target_transform = target_transform
        self.client_ids = sorted(list(self._h5_file['examples'].keys()))

    @property
    def training_file(self):
        return os.path.join(self._root, self._fileprefix + '_train.h5')

    @property
    def test_file(self):
        return os.path.join(self._root, self._fileprefix + '_test.h5')

    def _check_exists(self):
        return (os.path.exists(self.training_file) and
                os.path.exists(self.test_file))

    def download(self):
        if self._check_exists():
            return

        download_and_extract_archive(
            self._BASE_URL + self._fileprefix + '.tar.bz2',
            download_root=self._root,
            remove_finished=True
        )

    def create_dataset(self, client_id):
        client_h5 = self._h5_file[self._EXAMPLE_GROUP][client_id]
        return FEMNISTClientDataset(client_h5,
                                    transform=self._transform,
                                    target_transform=self._target_transform)


class FEMNISTDatasetPartition(torch.utils.data.ConcatDataset):

    def __init__(self, client_ids, datasets):
        super(FEMNISTDatasetPartition, self).__init__(datasets)
        self.client_ids = client_ids


class FEMNISTDatasetPartitioner(object):

    def __init__(self, dataset, num_splits, seed=None, max_partition_len=None):
        self._dataset = dataset
        self._partitions = []

        if num_splits < 1:
            raise ValueError('number of splits should be > 0')

        ids = list(dataset.client_ids)
        rng = random.Random()
        rng.seed(seed)
        rng.shuffle(ids)

        partition_len = int(len(ids) / num_splits)
        for _ in range(num_splits):
            self._partitions.append(ids[0:partition_len])
            ids = ids[partition_len:]
        # append remains to the last partition
        self._partitions[-1] += ids

        if max_partition_len:
            for partition in self._partitions:
                del partition[max_partition_len:]

    def __len__(self):
        return len(self._partitions)

    def get(self, idx):
        datasets = []
        for client_id in self._partitions[idx]:
            datasets.append(self._dataset.create_dataset(client_id))

        return FEMNISTDatasetPartition(self._partitions[idx], datasets)
