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


def main():
    num_trials = 1
    num_data = 3
    device = torch.device('cpu')

    weights = [1/num_data] * num_data
    max_iter = 15

    for _ in range(num_trials):
        models = generate_models(num_data, device)
        workers = [ADMMWorker(model, device) for model in models]
        mean = fedavg(models, weights=weights)
        aggregator_base = ADMMAggregator(workers, weights,
                                         max_iter=max_iter, threshold=0.0,
                                         lr=None,
                                         decay_rate=None, decay_period=None)

        n_rows = 3
        n_cols = num_data
        xs = list(range(max_iter))

        zs_axes = [plt.subplot(n_rows, n_cols, i) for i in range(1, num_data+1)]
        xs_axes = [plt.subplot(n_rows, n_cols, n_cols + i)
                   for i in range(1, num_data+1)]
        lambdas_axes = [plt.subplot(n_rows, n_cols, 2 * n_cols + i)
                        for i in range(1, num_data+1)]

        lrs = [4, 2, 1, 0.5]
        colors = COLORS[:len(lrs)]

        print(get_value(mean))

        for lr, color in zip(lrs, colors):
            aggregator = copy.deepcopy(aggregator_base)
            aggregator.lr = lr

            xs_history_dict = {i: [] for i in range(num_data)}
            zs_history_dict = {i: [] for i in range(num_data)}
            lambda_history_dict = {i: [] for i in range(num_data)}

            for _ in range(max_iter):
                for i in range(num_data):
                    worker = aggregator.admm_workers[i]
                    lambda_history_dict[i].append(worker.lambdas[0].item())

                aggregator.run_step()
                for i in range(num_data):
                    worker = aggregator.admm_workers[i]
                    xs_history_dict[i].append(worker.xs[0].item())
                    zs_history_dict[i].append(get_value(worker.zs))

            for i in range(num_data):
                plt.subplot(zs_axes[i])
                # plot zs
                plt.hlines(get_value(mean), 0, max_iter-1)
                plt.plot(xs, zs_history_dict[i],
                         color=color, label='rho=%s' % str(lr))
                if i == 0:
                    plt.ylabel('z')
                plt.legend()

                plt.subplot(xs_axes[i])
                # plot xs
                w = get_value(models[i].state_dict())
                plt.hlines(w, 0, max_iter-1)
                plt.plot(xs, xs_history_dict[i],
                         color=color, label='rho=%s' % str(lr))

                if i == 0:
                    plt.ylabel('x')
                plt.legend()

                plt.subplot(lambdas_axes[i])
                # plot lambdas
                plt.plot(xs, lambda_history_dict[i],
                         color=color, label='rho=%s' % str(lr))
                if i == 0:
                    plt.ylabel('lambda')
                plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
