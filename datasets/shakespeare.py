from __future__ import absolute_import, division, print_function

import h5py
import os.path
import torch.utils.data

from datasets.utils import download_and_extract_archive


class ShakespeareClientDataset(torch.utils.data.Dataset):

    def __init__(self, client_h5, client_id, transform=None):
        self._snippets = client_h5['snippets']
        self._transform = transform

    def __len__(self):
        return len(self._snippets)

    def __getitem__(self, index):
        data = self._snippets[index]

        if self._transform:
            data = self._transform(data)

        return data


class ShakespeareDataset(object):

    _EXAMPLE_GROUP = 'examples'
    _URL = 'https://storage.googleapis.com/tff-datasets-public/shakespeare.tar.bz2'

    def __init__(self, root, train=True, download=False, transform=None):
        self._root = root

        if download:
            self.download()

        if train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self._h5_file = h5py.File(data_file, 'r')
        self._transform = transform
        self.client_ids = sorted(list(self._h5_file['examples'].keys()))

    @property
    def training_file(self):
        return os.path.join(self._root, 'shakespeare_train.h5')

    @property
    def test_file(self):
        return os.path.join(self._root, 'shakespeare_test.h5')

    def _check_exists(self):
        return (os.path.exists(self.training_file) and
                os.path.exists(self.test_file))

    def download(self):
        if self._check_exists():
            return

        download_and_extract_archive(
            self._URL,
            download_root=self._root,
            remove_finished=False)

    def create_dataset(self, client_id):
        client_h5 = self._h5_file[self._EXAMPLE_GROUP][client_id]
        return ShakespeareClientDataset(client_h5, client_id,
                                        transform=self._transform)
