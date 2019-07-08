from __future__ import absolute_import, division, print_function

import torch
import torch.distributed as dist

from utils import logger
from utils.test import test_model

_LOGGER = logger.get_logger(__file__)


def dist_average(tensors):
    handles = []
    for tensor in tensors:
        handles.append(dist.all_reduce(
            tensor, op=dist.ReduceOp.SUM, async_op=True))

    with torch.no_grad():
        for i, tensor in enumerate(tensors):
            handles[i].wait()
            tensor.data /= dist.get_world_size()


class ADMMAverageCalculator(object):

    def __init__(self, max_iter, tolerance, lr):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.lr = lr
        self.total_iter = 0

    def run(self, tensors):
        with torch.no_grad():
            lr = self.lr
            lambdas = [torch.rand(tensor.shape) for tensor in tensors]
            zs = [torch.zeros(tensor.shape) for tensor in tensors]

            for i in range(self.max_iter):
                xs = [(1 / (2 + lr) * (2 * tensor - l + 2 * lr * z))
                      for tensor, l, z in zip(tensors, lambdas, zs)]
                zs = [x + l / lr for x, l in zip(xs, lambdas)]

                dist_average(zs)

                for l, x, z in zip(lambdas, xs, zs):
                    l += lr * (x - z)

                if i > 0:
                    distance = self._calculate_distance(zs, prev_zs)
                    _LOGGER.debug('Distance: %s', str(distance))
                    if distance < self.tolerance:
                        _LOGGER.debug('Average has converged at iter:%d', i)
                        break

                prev_zs = zs
                if i % 2 == 0:
                    lr /= 2
                self.total_iter += 1

            return zs

    def _calculate_distance(self, zs, prev_zs):
        distance = 0.0
        for z, prev_z in zip(zs, prev_zs):
            distance += torch.norm(z - prev_z)
        return distance


class Worker(object):

    def __init__(self, rank, num_workers, init_method,
                 backend='gloo', admm_kwargs=None):
        self.rank = rank
        self.num_workers = num_workers
        self.admm_avg_calculator = ADMMAverageCalculator(
            **admm_kwargs) if admm_kwargs else None

        dist.init_process_group(backend, init_method=init_method, rank=rank,
                                world_size=num_workers)

    def run(self, epochs, local_epochs, train_args, test_args=None, without_sync=False):
        # CAVEATS: assume that model parameters of all workers are the same at the beginning.
        # TODO: is this assumption necessary?

        for epoch in range(epochs):
            log_prefix = '[worker] rank: {}, epoch: [{}/{}]'.format(
                self.rank, epoch, epochs)
            for local_epoch in range(local_epochs):
                new_log_prefix = '{}, local_epoch: [{}/{}]'.format(
                    log_prefix, local_epoch, local_epochs)
                train_args.train_fn(train_args, log_prefix=new_log_prefix)

            if not without_sync:
                parameters = list(train_args.model.parameters())
                if self.admm_avg_calculator:
                    avgs = self.admm_avg_calculator.run(parameters)
                    for parameter, avg in zip(train_args.model.parameters(), avgs):
                        parameter.data = avg
                else:
                    dist_average(parameters)

            if test_args and epoch % test_args.period == 0:
                test_model(test_args, log_prefix)

        if self.rank == 0 and self.admm_avg_calculator:
            _LOGGER.info('Avg ADMM iteration: %s',
                         self.admm_avg_calculator.total_iter/epochs)
