from __future__ import absolute_import, division, print_function

import torch
import numpy as np
import functools

from sequential.admm import ADMMWorker, ADMMAggregator
from utils.test import test_model
import utils.logger as logger
import utils.ops as ops


_LOGGER = logger.get_logger(__file__, logger.INFO)


class Worker(object):

    def __init__(self, rank, local_epochs, train_args, test_args, writer=None):
        self.rank = rank
        self.local_epochs = local_epochs
        self.train_args = train_args
        self.test_args = test_args
        self.writer = writer
        self.metrics_list = None
        self.global_step = 0

    def train(self, log_prefix):
        self.metrics_list = []
        for local_epoch in range(1, self.local_epochs + 1):
            new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                log_prefix, local_epoch, self.local_epochs)
            metrics = self.train_fn(log_prefix=new_log_prefix)
            self.global_step += metrics['count']
            self.metrics_list.append(metrics)

    def test(self, log_prefix):
        return test_model(self.test_args, log_prefix)

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

    if weights is None:
        weights = [1 / len(models)] * len(models)

    return {name: ops.weighted_sum(tensors, weights)
            for name, tensors in aggregated_state_dict.items()}


def aggregate_models(workers, weights=None, admm_kwargs=None):
    if weights is None:
        weights = [1 / len(workers)] * len(workers)

    if admm_kwargs:
        admm_workers = [ADMMWorker(worker.model, worker.device)
                        for worker in workers]
        admm_aggregator = ADMMAggregator(admm_workers,
                                         weights, **admm_kwargs)
        admm_aggregator.run()
        return admm_aggregator.zs
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
