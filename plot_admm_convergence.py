from __future__ import absolute_import, division, print_function

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from sequential.admm import ADMMWorker, ADMMAggregator
from sequential.worker import fedavg
from utils.mock import MockModel
from utils.admm_parameter_tuner import ADMMParameterTuner


def generate_models(num, device, dim=1):
    models = []
    for _ in range(num):
        state_dict = {'weight': torch.rand(dim)}
        models.append(MockModel(state_dict, device))
    return models


def get_value(state_dict):
    return state_dict['weight'].item()


COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def run_admm_and_plot(aggregator_base, attr_name, attr_values, max_iter, mean):
    num_workers = len(aggregator_base.admm_workers)
    n_rows = 3
    n_cols = num_workers
    xs = list(range(max_iter))

    zs_axes = [plt.subplot(n_rows, n_cols, i) for i in range(1, num_workers+1)]
    xs_axes = [plt.subplot(n_rows, n_cols, n_cols + i)
               for i in range(1, num_workers+1)]
    lambdas_axes = [plt.subplot(n_rows, n_cols, 2 * n_cols + i)
                    for i in range(1, num_workers+1)]

    for attr_value, color in zip(attr_values, COLORS[:len(attr_values)]):
        aggregator = copy.deepcopy(aggregator_base)
        setattr(aggregator, attr_name, attr_value)

        xs_history_dict = {i: [] for i in range(num_workers)}
        zs_history_dict = {i: [] for i in range(num_workers)}
        lambda_history_dict = {i: [] for i in range(num_workers)}

        for _ in range(max_iter):
            for i in range(num_workers):
                worker = aggregator.admm_workers[i]
                lambda_history_dict[i].append(worker.lambdas[0].item())

            aggregator.run_step()
            for i in range(num_workers):
                worker = aggregator.admm_workers[i]
                xs_history_dict[i].append(worker.xs[0].item())
                zs_history_dict[i].append(get_value(worker.zs))

        for i in range(num_workers):
            plt.subplot(zs_axes[i])
            # plot zs
            plt.hlines(mean, 0, max_iter-1)
            plt.plot(xs, zs_history_dict[i],
                     color=color, label='%s=%s' % (attr_name, str(attr_value)))
            if i == 0:
                plt.ylabel('z')
            plt.legend()

            plt.subplot(xs_axes[i])
            # plot xs
            w = get_value(aggregator.admm_workers[i].model.state_dict())
            plt.hlines(w, 0, max_iter-1)
            plt.plot(xs, xs_history_dict[i],
                     color=color, label='%s=%s' % (attr_name, str(attr_value)))

            if i == 0:
                plt.ylabel('x')
            plt.legend()

            plt.subplot(lambdas_axes[i])
            # plot lambdas
            plt.plot(xs, lambda_history_dict[i],
                     color=color, label='%s=%s' % (attr_name, str(attr_value)))
            if i == 0:
                plt.ylabel('lambda')
            plt.legend()

    plt.show()


def main():
    num_workers = 3
    device = torch.device('cpu')

    weights = [1/num_workers] * num_workers
    max_iter = 15

    models = generate_models(num_workers, device)
    workers = [ADMMWorker(model, device) for model in models]
    mean = get_value(fedavg(models, weights=weights))
    print('Mean:', mean)

    aggregator_base = ADMMAggregator(workers, weights,
                                     max_iter=max_iter, threshold=0.0,
                                     lr=None,
                                     decay_rate=None, decay_period=None)

    ##### Different decay rates #####
    # aggregator_base.lr = 2
    # aggregator_base.decay_period = 1
    # decay_rates = [1, 0.8, 0.5, 0.3]
    # run_admm_and_plot(aggregator_base, 'decay_rate',
    #                   decay_rates, max_iter, mean)

    ##### Different lrs #####
    lrs = [4, 2, 1, 0.5]
    run_admm_and_plot(aggregator_base, 'lr', lrs, max_iter, mean)


if __name__ == "__main__":
    main()
