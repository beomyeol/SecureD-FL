from __future__ import absolute_import, division, print_function

import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10

from collections import namedtuple
from federated_dataset import FederatedDataset


DatasetAttr = namedtuple(
    'DatasetAttr', ['cls', 'transform'])


class ImageToNumpy(object):

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        return self.to_tensor(img).numpy()


_DATASET_ATTR = {
    "cifar10": DatasetAttr(CIFAR10, transform=ImageToNumpy()),
}


def get_dataset_attr(dataset_name):
    return _DATASET_ATTR[dataset_name]


def split_dataset(dataset, num_clients, seed):
    """Split dataset into multiple chunks.

    This assumes that the dataset returns (data, label) tuples.
    """
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
        xs = np.stack(xs)
        ys = np.stack(ys)

        dataset_splits.append(('u_%d' % i, xs, ys, split))

    return dataset_splits


class FederatedClientDataset(torch.utils.data.Dataset):

    def __init__(self, client_id, X, Y, indexes):
        assert len(X) == len(Y), "Different size"
        self.client_id = client_id
        self.X = X
        self.Y = Y
        # indexes in the original dataset
        self.indexs = indexes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index]


class FederatedDatasetImpl(FederatedDataset):

    def __init__(self, dataset, num_clients, seed):
        self._seed = seed
        self._client_datasets = {
            split[0]: FederatedClientDataset(*split)
            for split in split_dataset(dataset, num_clients, seed)
        }
        self._client_ids = list(self._client_datasets.keys())

    def client_ids(self):
        return self._client_ids

    def get_client_dataset(self, client_id):
        return self._client_datasets[client_id]


def create_dataset(dataset_name, root_dir, train, num_clients, seed, download):
    attr = get_dataset_attr(dataset_name)
    dataset = attr.cls(
        root_dir,
        train=train,
        download=download,
        transform=attr.transform)
    return FederatedDatasetImpl(dataset, num_clients, seed)
