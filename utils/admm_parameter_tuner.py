from __future__ import absolute_import, division, print_function

import collections
import copy
import heapq
import itertools

import utils.logger as logger
import utils.ops as ops
from sequential.admm import ADMMAggregator, ADMMWorker
from sequential.worker import fedavg

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
                 thresholds, max_iters, weights=None,
                 early_stop_threshold=None):
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
        self.early_stop_threshold = early_stop_threshold

        self.results = []
        self.means = fedavg(models, self.weights)

    def run(self):
        self.results = []
        admm_params_tuple_list = itertools.product(
            self.lrs, self.decay_periods, self.decay_rates, self.thresholds,
            self.max_iters)
        for admm_params_tuple in admm_params_tuple_list:
            lr = admm_params_tuple[0]
            decay_period = admm_params_tuple[1]
            decay_rate = admm_params_tuple[2]
            threshold = admm_params_tuple[3]
            max_iter = admm_params_tuple[4]
            admm_params = ADMMParameters(
                lr=lr,
                decay_period=decay_period,
                decay_rate=decay_rate,
                threshold=threshold,
                max_iter=max_iter)
            self.results.append(self._run_admm(admm_params))

    def _run_admm(self, admm_params):
        _LOGGER.debug('ADMM parameters: %s', str(admm_params))

        admm_aggregator = ADMMAggregator(
            copy.deepcopy(self.admm_workers),
            self.weights,
            max_iter=admm_params.max_iter,
            threshold=admm_params.threshold,
            lr=admm_params.lr,
            decay_period=admm_params.decay_period,
            decay_rate=admm_params.decay_rate)

        if self.early_stop_threshold:
            for _ in range(admm_aggregator.max_iter):
                admm_aggregator.run_step()
                mse = ops.calculate_mse(admm_aggregator.zs, self.means).item()
                if mse < self.early_stop_threshold:
                    break
        else:
            admm_aggregator.run()
            mse = ops.calculate_mse(admm_aggregator.zs, self.means).item()

        _LOGGER.debug('ADMM finished. iter=%d, mse=%s',
                      admm_aggregator.current_iter, str(mse))

        return ADMMTuneResult(
            iter=admm_aggregator.current_iter,
            mse=mse,
            parameters=admm_params,
            state_dicts=admm_aggregator.zs_history)

    def get(self, n=1, key=('iter', 'mse')):
        return heapq.nsmallest(n, self.results,
                               key=lambda x: [getattr(x, name)
                                              for name in key])
