from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import functools

from utils.test import test_model
import utils.logger as logger


_LOGGER = logger.get_logger(__file__)


class ADMMAggregator(object):

    def __init__(self, model):
        self.model = model
        self.lambdas = [torch.rand(parameter.shape)
                        for parameter in model.parameters()]
        self.zs = {name: torch.zeros(parameter.shape)
                   for name, parameter in model.named_parameters()}
        self.xs = None

    def update(self, lr):
        with torch.no_grad():
            self.xs = [(1 / (2 + lr) * (2 * param - l + 2 * lr * z))
                       for param, l, z
                       in zip(self.model.parameters(),
                              self.lambdas,
                              self.zs.values())]
            self.zs = {name: x + l / lr
                       for name, x, l in zip(self.zs, self.xs, self.lambdas)}

    def update_lambdas(self, lr):
        with torch.no_grad():
            zs = self.zs.values()
            for l, x, z in zip(self.lambdas, self.xs, zs):
                l += lr * (x - z)


class Worker(object):

    def __init__(self, rank, local_epochs, train_args, test_args):
        self.rank = rank
        self.local_epochs = local_epochs
        self.train_args = train_args
        self.test_args = test_args
        self.losses = None

    def train(self, log_prefix):
        self.losses = []
        for local_epoch in range(1, self.local_epochs + 1):
            new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                log_prefix, local_epoch, self.local_epochs)
            self.losses += self.train_fn(log_prefix=new_log_prefix)

    def test(self, log_prefix):
        test_model(self.test_args, log_prefix)

    @property
    def model(self):
        return self.train_args.model

    @property
    def train_fn(self):
        return functools.partial(self.train_args.train_fn, self.train_args)

    def __repr__(self):
        return '<worker rank=%d>' % self.rank


def _weighted_sum(tensors, weights):
    with torch.no_grad():
        weighted_tensors = torch.stack([
            weight * tensor
            for weight, tensor in zip(weights, tensors)])
        return torch.sum(weighted_tensors, dim=0)


def _calculate_distance(zs, prev_zs):
    distance = 0.0
    with torch.no_grad():
        for z, prev_z in zip(zs, prev_zs):
            distance += torch.norm(z - prev_z)
    return distance


def _run_admm_aggregation(aggregators, weights, max_iter, tolerance, lr):
    current_lr = lr
    zs = None

    for i in range(max_iter):
        zs_list_dict = {}
        for aggregator in aggregators:
            aggregator.update(current_lr)
            if zs_list_dict:
                for name, z in aggregator.zs.items():
                    zs_list_dict[name].append(z)
            else:
                zs_list_dict = {name: [z]
                                for name, z in aggregator.zs.items()}

        zs = {name: _weighted_sum(zs_list, weights)
              for name, zs_list in zs_list_dict.items()}

        for aggregator in aggregators:
            aggregator.zs = zs
            aggregator.update_lambdas(current_lr)

        if i > 0:
            distance = _calculate_distance(zs.values(), prev_zs)
            _LOGGER.debug('Distance: %s', str(distance.item()))
            if distance < tolerance:
                _LOGGER.info('ADMM aggregation has converged at iter: %d', i)
                break

        prev_zs = zs.values()
        if i % 2 == 0:
            current_lr /= 2

    return zs


def aggregate_models(workers, weights=None, admm_kwargs=None):
    if weights is None:
        weights = [1 / len(workers)] * len(workers)

    if admm_kwargs:
        admm_aggregators = [ADMMAggregator(worker.model)
                            for worker in workers]
        return _run_admm_aggregation(admm_aggregators,
                                     weights,
                                     **admm_kwargs)
    else:
        tensor_list_dict = {}
        for worker in workers:
            for name, parameter in worker.model.named_parameters():
                if name in tensor_list_dict:
                    tensor_list_dict[name].append(parameter)
                else:
                    tensor_list_dict[name] = [parameter]

        return {name: _weighted_sum(tensors, weights)
                for name, tensors in tensor_list_dict.items()}


def run_clustering(workers, num_clusters):
    from sklearn.cluster import KMeans
    # generate inputs

    X = []
    for worker in workers:
        flattened_parameters = [parameter.data.numpy().flatten()
                                for parameter in worker.model.parameters()]
        X.append(np.concatenate(flattened_parameters, axis=None))

    kmeans = KMeans(n_clusters=num_clusters).fit(X)
    return kmeans
