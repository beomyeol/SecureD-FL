from __future__ import absolute_import, division, print_function

import torch
import torch.utils.data
import random


class DatasetPartition(torch.utils.data.ConcatDataset):

    def __init__(self, client_ids, datasets):
        super(DatasetPartition, self).__init__(datasets)
        self.client_ids = client_ids


class DatasetPartitioner(object):

    def __init__(self, dataset, num_splits,
                 ratios=None, seed=None, max_partition_len=None):
        self._dataset = dataset
        self._partitions = []

        if num_splits < 1:
            raise ValueError('number of splits should be > 0')

        num_client_ids = len(dataset.client_ids())
        if num_splits > num_client_ids:
            raise ValueError('#splits (%d) should be <= #client ids (%d)' % (
                num_splits, num_client_ids))

        if ratios:
            if type(ratios) == str:
                ratios = [float(ratio) for ratio in ratios.split(',')]

            if len(ratios) != num_splits:
                raise ValueError('invalid length of ratios')

        ids = list(dataset.client_ids())
        rng = random.Random()
        rng.seed(seed)
        rng.shuffle(ids)
        num_clients = len(ids)
        partition_len = int(num_clients / num_splits)

        for i in range(num_splits):
            if ratios:
                partition_len = int(num_clients * ratios[i])
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
            datasets.append(self._dataset.get_client_dataset(client_id))

        return DatasetPartition(self._partitions[idx], datasets)


def get_partition(dataset, rank, world_size, seed, ratios=None,
                  max_num_users_per_worker=None):
    partitioner = DatasetPartitioner(dataset,
                                     world_size,
                                     ratios,
                                     seed,
                                     max_num_users_per_worker)
    return partitioner.get(rank)
