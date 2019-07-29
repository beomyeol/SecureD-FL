from __future__ import absolute_import, division, print_function

import h5py
import torch.utils.data
import os.path
import torch

import datasets.partition as partition


class VehicleClientDataset(torch.utils.data.Dataset):

    def __init__(self, client_h5, transform=None, target_transform=None):
        self._values = client_h5['values']
        self._labels = client_h5['labels']
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        value = self._values[index, :]
        label = torch.FloatTensor([self._labels[index].item()])

        if self._transform:
            value = self._transform(value)

        if self._target_transform:
            label = self._target_transform(label)

        return value, label


class VehicleDataset(object):

    _EXAMPLE_GROUP = 'examples'
    _GDRIVE_URL = 'https://drive.google.com/file/d/1x39M3wYLt0VKNS8RSiwohT-VfVoGeKlM/view?usp=sharing'
    _INTERNAL_URL = 'http://gofile.me/41raf/y8dqxtbVr'

    def __init__(self, root, train=True, download=False, transform=None,
                 target_transform=None):
        self._root = root

        if download:
            raise RuntimeError('Downloading vehicle dataset is not supported. '
                               'Please download raw file from %s or %s, and '
                               'run scripts/preprocess_vehicle.py' %
                               (self._GDRIVE_URL, self._INTERNAL_URL))
        if train:
            data_file = self.train_file
        else:
            data_file = self.test_file

        self._h5_file = h5py.File(data_file, 'r')
        self._transform = transform
        self._target_transform = target_transform
        self.client_ids = sorted(list(self._h5_file['examples'].keys()))

    @property
    def train_file(self):
        return os.path.join(self._root, 'vehicle_train.h5')

    @property
    def test_file(self):
        return os.path.join(self._root, 'vehicle_test.h5')

    def create_dataset(self, client_id):
        client_h5 = self._h5_file[self._EXAMPLE_GROUP][client_id]
        return VehicleClientDataset(client_h5,
                                    transform=self._transform,
                                    target_transform=self._target_transform)


def load_dataset(dataset_dir, train=True, dataset_download=False, **kwargs):
    def transform(x):
        return torch.from_numpy(x).float()
    return VehicleDataset(dataset_dir,
                          train=train,
                          download=dataset_download,
                          transform=transform)
