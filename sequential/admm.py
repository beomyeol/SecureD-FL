from __future__ import absolute_import, division, print_function

import torch
from operator import itemgetter

import utils.logger as logger
import utils.ops as ops

_LOGGER = logger.get_logger(__file__, logger.INFO)


class ADMMWorker(object):

    def __init__(self, model, device):
        self.model = model
        self.lambdas = [torch.rand(parameter.shape).to(device)
                        for parameter in model.parameters()]
        self.zs = {name: torch.zeros(parameter.shape).to(device)
                   for name, parameter in model.named_parameters()}
        self.xs = None

    def update(self, lr):
        with torch.no_grad():
            self.xs = [(1 / (2 + lr) * (2 * param - l + lr * z))
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


class ADMMAggregator(object):

    def __init__(self, admm_workers, weights, max_iter, threshold, lr,
                 decay_period, decay_rate, groups_pair=None):
        self.admm_workers = admm_workers
        self.weights = weights
        self.max_iter = max_iter
        self.threshold = threshold
        self.lr = lr
        self.decay_period = decay_period
        self.decay_rate = decay_rate
        self.groups_pair = groups_pair
        self._current_iter = 0

        # Stats
        self.zs_history = []

    @property
    def current_iter(self):
        return self._current_iter

    @property
    def zs(self):
        if self.zs_history:
            return self.zs_history[-1]
        else:
            return None

    def is_converged(self):
        if len(self.zs_history) > 2:
            distance = ops.calculate_distance(self.zs_history[-1].values(),
                                              self.zs_history[-2].values())
            _LOGGER.debug('ADMM Z Distance: %s', str(distance.item()))
            return distance < self.threshold
        else:
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

        if self.groups_pair:
            groups = self.groups_pair[self._current_iter %
                                      len(self.groups_pair)]
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
                _LOGGER.info(
                    'ADMM aggregation has ended at iter: %d', self.current_iter)
                break
        return self.is_converged()
