from __future__ import absolute_import, division, print_function

import heapq
import collections
import copy

from sequential.worker import fedavg
from sequential.admm import ADMMWorker, ADMMAggregator
import utils.logger as logger
import utils.ops as ops

_LOGGER = logger.get_logger(__file__)

ADMMParameters = collections.namedtuple(
    'ADMMParameters',
    [
        'lr',
        'decay_rate',
        'decay_period',
        'threshold',
        'max_iter',
    ]
)

ADMMTuneResult = collections.namedtuple(
    'ADMMTuneResult',
    [
        'iter',
        'mse',
        'parameters',
        'state_dicts',
    ]
)


class ADMMParameterTuner(object):

    def __init__(self, models, device, lrs, decay_rates, decay_periods,
                 thresholds, max_iters, weights=None):
        self.admm_workers = [ADMMWorker(model, device) for model in models]
        self.lrs = lrs
        self.decay_rates = decay_rates
        self.decay_periods = decay_periods
        self.thresholds = thresholds
        self.max_iters = max_iters
        if weights:
            self.weights = weights
        else:
            self.weights = [1/len(models)] * len(models)
        self.results = []
        self.means = fedavg(models, self.weights)

    def run(self):
        self.results = []
        for lr in self.lrs:
            for decay_period in self.decay_periods:
                for decay_rate in self.decay_rates:
                    for threshold in self.thresholds:
                        for max_iter in self.max_iters:
                            admm_params = ADMMParameters(
                                lr=lr,
                                decay_period=decay_period,
                                decay_rate=decay_rate,
                                threshold=threshold,
                                max_iter=max_iter)
                            self.results.append(self._run_admm(admm_params))

    def _run_admm(self, admm_params):
        admm_aggregator = ADMMAggregator(copy.deepcopy(self.admm_workers),
                                         self.weights,
                                         admm_params.max_iter,
                                         admm_params.threshold,
                                         admm_params.lr,
                                         admm_params.decay_period,
                                         admm_params.decay_rate)
        admm_aggregator.run()
        return ADMMTuneResult(
            iter=admm_aggregator.current_iter,
            mse=ops.calculate_mse(admm_aggregator.zs, self.means).item(),
            parameters=admm_params,
            state_dicts=admm_aggregator.zs_history)

    def get(self, n=1, key=('iter', 'mse')):
        return heapq.nsmallest(n, self.results,
                               key=lambda x: [getattr(x, name)
                                              for name in key])
