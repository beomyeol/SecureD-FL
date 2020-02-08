from __future__ import absolute_import, division, print_function

import utils.logger as logger
from datasets import dataset_factory

_LOGGER = logger.get_logger(__file__)


def load_dataset(dataset_dir, seed, train=True, dataset_download=False,
                 num_clients=1000, **kwargs):
    return dataset_factory.create_dataset(
        'cifar10', root_dir=dataset_dir, train=train, num_clients=num_clients,
        seed=seed, download=dataset_download)
