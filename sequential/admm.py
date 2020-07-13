"""ADMM aggregation."""
# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=invalid-name
from __future__ import absolute_import, division, print_function

from operator import itemgetter

import torch

import utils.logger as logger
import utils.ops as ops

_LOGGER = logger.get_logger(__file__, logger.INFO)


class ADMMWorker():

    def __init__(self, model, device, rho_gen_fn=None, record_zs_history=False):
        self.model = model
        self.lambdas = [torch.rand(parameter.shape, device=device)
                        for parameter in model.parameters()]
        self.zs = {name: torch.zeros(parameter.shape, device=device)
                   for name, parameter in model.named_parameters()}
        self.xs = None
        self.rho_gen_fn = rho_gen_fn
        self.zs_history = [] if record_zs_history else None

    def update(self, lr):
        with torch.no_grad():
            if self.rho_gen_fn:
                lr = self.rho_gen_fn(lr)
            self.xs = [(1 / (2 + lr) * (2 * param - l + lr * z))
                       for param, l, z
                       in zip(self.model.parameters(),
                              self.lambdas,
                              self.zs.values())]
            self.zs = {name: x + l / lr
                       for name, x, l in zip(self.zs, self.xs, self.lambdas)}
            if self.zs_history is not None:
                self.zs_history.append(self.zs)

    def update_lambdas(self, lr):
        with torch.no_grad():
            zs = self.zs.values()
            for l, x, z in zip(self.lambdas, self.xs, zs):
                l += lr * (x - z)


class ADMMAggregator():
    # pylint: disable=too-many-instance-attributes,too-many-arguments

    def __init__(self, admm_workers, weights, max_iter, lr, decay_period,
                 decay_rate, threshold=None, groups=None):
        self.admm_workers = admm_workers
        self.weights = weights
        self.max_iter = max_iter
        self.lr = lr
        self.decay_period = decay_period
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.groups = groups

        self._current_iter = 0

        # Stats
        self.zs_history = []

        # limit max iteration when groups is provided (secure admm)
        if self.groups:
            iter_limit = 2 * len(self.groups) - 1  # gap: len(self.groups)
            if self.max_iter > iter_limit:
                _LOGGER.info('Max iteration changed from %d to %d',
                             self.max_iter, iter_limit)
                self.max_iter = iter_limit

    @property
    def current_iter(self):
        return self._current_iter

    @property
    def zs(self):
        if self.zs_history:
            return self.zs_history[-1]

        return None

    def is_converged(self):
        if self.threshold and len(self.zs_history) > 2:
            distance = ops.calculate_distance(self.zs_history[-1].values(),
                                              self.zs_history[-2].values())
            _LOGGER.debug('ADMM Z Distance: %s', str(distance.item()))
            return distance < self.threshold

        return False

    def _calculate_weighted_zs_sum(self):
        def weighted_zs_sum(indices=None):
            if indices:
                target_workers = itemgetter(*indices)(self.admm_workers)
                target_weights = itemgetter(*indices)(self.weights)
            else:
                target_workers = self.admm_workers
                target_weights = self.weights

            aggregated_zs_dict = ops.aggregate_state_dicts_by_names(
                [admm_worker.zs for admm_worker in target_workers])
            return {name: ops.weighted_sum(zs_list, target_weights)
                    for name, zs_list in aggregated_zs_dict.items()}

        if self.groups:
            groups = self.groups[self._current_iter % len(self.groups)]
            intermediate_zs_dicts = [weighted_zs_sum(group)
                                     for group in groups]
            aggregated_zs_dict = ops.aggregate_state_dicts_by_names(
                intermediate_zs_dicts)

            with torch.no_grad():
                zs = {name: torch.sum(torch.stack(aggregated_zs), dim=0)
                      for name, aggregated_zs in aggregated_zs_dict.items()}
        else:
            zs = weighted_zs_sum()

        return zs

    def run_step(self):
        # x and z minimization
        for admm_worker in self.admm_workers:
            admm_worker.update(self.lr)

        zs = self._calculate_weighted_zs_sum()
        self.zs_history.append(zs)

        # lambda update
        for admm_worker in self.admm_workers:
            admm_worker.zs = zs
            admm_worker.update_lambdas(self.lr)

        self._current_iter += 1

        if self.decay_period and self.current_iter % self.decay_period == 0:
            self.lr *= self.decay_rate
            _LOGGER.debug('New LR: %s after iter %d',
                          str(self.lr), self.current_iter)

        return self.is_converged()

    def run(self):
        for _ in range(self.max_iter):
            if self.run_step():
                break
        _LOGGER.info('ADMM aggregation has ended at iter: %d',
                     self.current_iter)
        return self.is_converged()
