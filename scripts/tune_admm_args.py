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


def calculate_mse(tensors, others):
    flattened_tensors = torch.cat([t.flatten() for t in tensors])
    flattened_others = torch.cat([t.flatten() for t in others])
    return F.mse_loss(flattened_tensors, flattened_others)


def run_admm_aggregation(means, aggregators,
                         max_iter, threshold,
                         decay_rate, decay_period):
    weights = [1/len(aggregators)] * len(aggregators)
    zs = None

    for i in range(max_iter):
        zs_list_dict = {}
        for aggregator in aggregators:
            aggregator.update()
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
            aggregator.update_lambdas()

        if i > 0:
            distance = _calculate_distance(zs.values(), prev_zs)
            print('Distance: %s' % str(distance.item()))
            if distance < threshold:
                break

        mse = calculate_mse(zs.values(), means).item()
        print('MSE:', str(mse))

        prev_zs = zs.values()
        if i % decay_period == 0:
            for aggregator in aggregators:
                aggregator.lr *= decay_rate
            #print('lr:', str([aggregator.lr for aggregator in aggregators]))

    return zs, i+1, mse


def calculate_means(models):
    aggregated_state_dict = collections.OrderedDict()
    for model in models:
        for name, parameter in model.named_parameters():
            if name in aggregated_state_dict:
                aggregated_state_dict[name].append(parameter)
            else:
                aggregated_state_dict[name] = [parameter]

    weights = [1/len(models) * len(models)]

    return [_weighted_sum(parameters, weights)
            for parameters in aggregated_state_dict.values()]
    # with torch.no_grad():
        # return [torch.mean(torch.stack(parameters))
                # for parameters in aggregated_state_dict.values()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs=1, help='checkpoint path')

    args = parser.parse_args()

    device = torch.device('cpu')

    save_dict = torch.load(args.INPUT[0])
    models = [MockModel(state_dict)
              for state_dict in save_dict.values()]
    max_iter = 20
    threshold = 1e-5

    min_i = max_iter
    min_mse = None
    min_lr = None
    min_decay_rate = None
    min_decay_period = None

    means = calculate_means(models)

    for lr in [1, 8e-1, 5e-1, 1e-1]:#, 1e-2, 6e-3, 3e-3, 1e-3]:
        aggregators = [ADMMAggregator(model, device, lr)
                       for model in models]
        for decay_period in [1, 2, 4, 8]: 
            for decay_rate in [1, 0.9, 0.8, 0.5]:
                print('lr: {}, decay_period: {}, decay_rate: {}'.format(
                    lr, decay_period, decay_rate))
                sys.stdout.flush()


                iter_list, mse_list = [], []
                for _ in range(5):
                    zs, i, mse = run_admm_aggregation(
                        means, copy.deepcopy(aggregators),
                        max_iter, threshold,
                        decay_rate, decay_period)
                    print('iter: {}, mse: {}'.format(i, mse))
                    sys.stdout.flush()
                    iter_list.append(i)
                    mse_list.append(mse)

                i = np.mean(iter_list)
                mse = np.mean(mse)
                print("AVG iteration: {}, AVG mse: {}".format(i, mse))

                if i < min_i or (i == min_i and mse < min_mse):
                    min_i = i
                    min_mse = mse
                    min_lr = lr
                    min_decay_rate = decay_rate
                    min_decay_period = decay_period

    print('Min iter:', min_i)
    print('Min lr:', min_lr)
    print('Min mse:', min_mse)
    print('Min decay rate:', min_decay_rate)
    print('Min decay period:', min_decay_period)


if __name__ == "__main__":
    main()
