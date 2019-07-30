from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import functools
from operator import itemgetter

from utils.test import test_model
import utils.logger as logger
import utils.ops as ops


_LOGGER = logger.get_logger(__file__, logger.INFO)


class ADMMAggregator(object):

    def __init__(self, model, device):
        self.model = model
        self.lambdas = [torch.rand(parameter.shape).to(device)
                        for parameter in model.parameters()]
        self.zs = {name: torch.zeros(parameter.shape).to(device)
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

    @property
    def device(self):
        return self.train_args.device

    def __repr__(self):
        return '<worker rank=%d>' % self.rank


def _calculate_zs(aggregators, weights):
    aggregated_zs_dict = ops.aggregate_state_dicts_by_names(
        [aggregator.zs for aggregator in aggregators])
    return {name: ops.weighted_sum(zs_list, weights)
            for name, zs_list in aggregated_zs_dict.items()}


def run_admm_aggregation(aggregators, weights, max_iter, threshold, lr,
                         decay_period, decay_rate, groups_pair=None,
                         verbose=False):
    zs = None
    current_lr = lr
    distances = []

    for i in range(1, max_iter+1):
        for aggregator in aggregators:
            aggregator.update(current_lr)

        if groups_pair:
            groups = groups_pair[i % len(groups_pair)] if groups_pair else None
            intermediate_zs_dicts = []
            for group in groups:
                aggregator_group = itemgetter(*group)(aggregators)
                weight_group = itemgetter(*group)(weights)
                intermediate_zs_dicts.append(
                    _calculate_zs(aggregator_group, weight_group))

            aggregated_zs_dict = ops.aggregate_state_dicts_by_names(
                intermediate_zs_dicts)

            with torch.no_grad():
                zs = {name: torch.sum(torch.stack(aggregated_zs), dim=0)
                      for name, aggregated_zs in aggregated_zs_dict.items()}
        else:
            zs = _calculate_zs(aggregators, weights)

        for aggregator in aggregators:
            aggregator.zs = zs
            aggregator.update_lambdas(current_lr)

        if i > 1:
            distance = ops.calculate_distance(zs.values(), prev_zs)
            distances.append(distance)
            _LOGGER.debug('ADMM Z Distance: %s', str(distance.item()))
            if distance < threshold:
                break

        if decay_period and i % decay_period == 0:
            current_lr *= decay_rate
            _LOGGER.debug('New LR: %s after iter %d', str(current_lr), i)

        prev_zs = zs.values()

    _LOGGER.info('ADMM aggregation has ended at iter: %d', i)
    if verbose:
        return zs, i, distances
    else:
        return zs


def fedavg(models, weights=None):
    aggregated_state_dict = ops.aggregate_state_dicts_by_names(
        [model.state_dict() for model in models])

    return {name: ops.weighted_sum(tensors, weights)
            for name, tensors in aggregated_state_dict.items()}


def aggregate_models(workers, weights=None, admm_kwargs=None, verbose=False):
    if weights is None:
        weights = [1 / len(workers)] * len(workers)

    if admm_kwargs:
        admm_aggregators = [ADMMAggregator(worker.model, worker.device)
                            for worker in workers]
        retval = run_admm_aggregation(admm_aggregators,
                                      weights,
                                      verbose=verbose,
                                      **admm_kwargs)
        if verbose:
            models = [worker.model for worker in workers]
            avg = fedavg(models, weights)
            estimates = retval[0]
            _LOGGER.info('ADMM MSE: %s', str(
                calculate_mse(estimates, avg).item()))
        return retval
    else:
        models = [worker.model for worker in workers]
        return fedavg(models, weights)


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
