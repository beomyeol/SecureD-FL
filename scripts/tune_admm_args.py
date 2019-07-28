from __future__ import absolute_import, division, print_function

import argparse
import copy
import collections
import torch
import torch.nn.functional as F
import sys
import os.path
import numpy as np
import random

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from sequential.worker import ADMMAggregator, _run_admm_aggregation, _calculate_distance, fedavg, calculate_mse


class MockWorker(object):

    def __init__(self, model):
        self.model = model


class MockModel(object):

    def __init__(self, state_dict, device):
        self._state_dict = {name: parameter.to(device)
                            for name, parameter in state_dict.items()}

    def parameters(self):
        for parameter in self._state_dict.values():
            yield parameter

    def named_parameters(self):
        for name, parameter in self._state_dict.items():
            yield name, parameter

    def state_dict(self):
        return self._state_dict


def calculate_distance_z_and_param(aggregators):
    diffs = []
    for aggregator in aggregators:
        diffs.append(_calculate_distance(aggregator.model.state_dict(),
                                         aggregator.zs))
    retval = np.mean(diffs)
    print('Avg distance between the parameters and zs: ', str(retval))
    return retval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs=1, help='checkpoint path')

    args = parser.parse_args()

    device = torch.device('cpu')

    save_dict = torch.load(args.INPUT[0])
    models = [MockModel(state_dict, device)
              for state_dict in save_dict.values()]
    aggregators = [ADMMAggregator(model, device) for model in models]
    weights = [1/len(models)] * len(models)

    max_iter = 10
    threshold = 1e-8

    min_iter = max_iter
    min_mse = float('inf')
    min_lr = None
    min_decay_rate = None
    min_decay_period = None

    mean = fedavg([MockWorker(model) for model in models], weights)

    for lr in [1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4]:
        for decay_period in [1, 2, 4, 8]:
            for decay_rate in [1, 0.9, 0.8, 0.5]:
                print('lr: {}, decay_period: {}, decay_rate: {}'.format(
                    lr, decay_period, decay_rate))
                sys.stdout.flush()

                zs, iter, distance = _run_admm_aggregation(
                    copy.deepcopy(aggregators), weights, max_iter,
                    threshold, lr, decay_period, decay_rate, verbose=True)

                mse = calculate_mse(mean, zs).item()

                print('iter: {}, mse: {}'.format(iter, mse))
                sys.stdout.flush()

                if iter < min_iter or (iter == min_iter and mse < min_mse):
                    min_iter = iter
                    min_mse = mse
                    min_lr = lr
                    min_decay_rate = decay_rate
                    min_decay_period = decay_period

                print('Min iter:', min_iter)
                print('Min lr:', min_lr)
                print('Min mse:', min_mse)
                print('Min decay rate:', min_decay_rate)
                print('Min decay period:', min_decay_period)


if __name__ == "__main__":
    main()
