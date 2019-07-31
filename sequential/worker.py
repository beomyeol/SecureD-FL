from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import functools

from sequential.admm import ADMMAggregator, run_admm_aggregation
from utils.test import test_model
import utils.logger as logger
import utils.ops as ops


_LOGGER = logger.get_logger(__file__, logger.INFO)


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
            estimates = retval[0][-1]
            _LOGGER.info('ADMM MSE: %s', str(
                ops.calculate_mse(estimates, avg).item()))
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
