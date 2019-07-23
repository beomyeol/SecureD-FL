from __future__ import absolute_import, division, print_function

import argparse
import collections
import torch
import sys
import os.path
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from sequential.worker import ADMMAggregator, _calculate_distance, _weighted_sum


class MockModel(object):

    def __init__(self, state_dict):
        self.state_dict = state_dict

    def parameters(self):
        for parameter in self.state_dict.values():
            yield parameter

    def named_parameters(self):
        for name, parameter in self.state_dict.items():
            yield name, parameter


def calculate_distance_z_and_param(aggregators):
    diffs = []
    for aggregator in aggregators:
        diff = 0.0
        for name, parameter in aggregator.model.named_parameters():
            z = aggregator.zs[name]
            with torch.no_grad():
                diff += torch.norm(parameter - z).item()
        diffs.append(diff)
    retval = np.mean(diffs)
    print('Avg distance between the parameters and zs: ', str(retval))
    return retval


def run_admm_aggregation(means, aggregators,
                         max_iter, tolerance, lr,
                         decay_rate, decay_period):
    weights = [1/len(aggregators)] * len(aggregators)
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

        # calculate_distance_z_and_param(aggregators)

        zs = {name: _weighted_sum(zs_list, weights)
              for name, zs_list in zs_list_dict.items()}

        for aggregator in aggregators:
            aggregator.zs = zs
            aggregator.update_lambdas(current_lr)

        if i > 0:
            distance = _calculate_distance(zs.values(), prev_zs)
            #print('Distance: %s' % str(distance.item()))
            if distance < tolerance:
                #print('ADMM aggregation has converged at iter: %d' % i)
                break

        diff_with_means = _calculate_distance(zs.values(), means).item()
        # print('Distance between the estimate and the exact mean:',
        #      str(diff_with_means))

        prev_zs = zs.values()
        if i % decay_period == 0:
            current_lr *= decay_rate

    return zs, i, diff_with_means


def calculate_means(models):
    aggregated_state_dict = collections.OrderedDict()
    for model in models:
        for name, parameter in model.named_parameters():
            if name in aggregated_state_dict:
                aggregated_state_dict[name].append(parameter)
            else:
                aggregated_state_dict[name] = [parameter]

    with torch.no_grad():
        return [torch.mean(torch.stack(parameters))
                for parameters in aggregated_state_dict.values()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs=1, help='checkpoint path')

    args = parser.parse_args()

    save_dict = torch.load(args.INPUT[0])

    models = [MockModel(state_dict)
              for state_dict in save_dict.values()]

    means = calculate_means(models)

    aggregators = [ADMMAggregator(model, torch.device('cpu'))
                   for model in models]

    max_iter = 5
    tolerance = 1e-2

    min_i = max_iter
    min_diff_with_means = None
    min_lr = None
    min_decay_rate = None
    min_decay_period = None

    for lr in [0.3, 0.1, 0.06, 0.03, 0.01, 0.006, 0.003, 0.001]:
        for decay_period in [1, 2, 3, 4]:
            for decay_rate in [1, 0.8, 0.5, 0.3, 0.1]:
                print('lr: {}, decay_period: {}, decay_rate: {}'.format(
                    lr, decay_period, decay_rate))
                zs, i, diff_with_means = run_admm_aggregation(
                    means, aggregators,
                    max_iter, tolerance, lr,
                    decay_rate, decay_period)
                print('iter: {}, distance: {}'.format(i, diff_with_means))

                if i < min_i or (i == min_i and diff_with_means < min_diff_with_means):
                    min_i = i
                    min_diff_with_means = diff_with_means
                    min_lr = lr
                    min_decay_rate = decay_rate
                    min_decay_period = decay_period

    print('Min iter:', min_i)
    print('Min distance:', min_diff_with_means)
    print('Min decay rate:', min_decay_rate)
    print('Min decay period:', min_decay_period)


if __name__ == "__main__":
    main()
