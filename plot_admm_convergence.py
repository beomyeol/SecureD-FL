from __future__ import absolute_import, division, print_function

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


def main():
    num_trials = 1
    num_data = 2
    device = torch.device('cpu')

    weights = [1/num_data] * num_data
    max_iter = 15

    for _ in range(num_trials):
        models = generate_models(num_data, device)
        workers = [ADMMWorker(model, device) for model in models]
        mean = fedavg(models, weights=weights)
        aggregator = ADMMAggregator(workers, weights,
                                    max_iter=max_iter, threshold=0.0, lr=2,
                                    decay_rate=None, decay_period=None)

        xs_history_dict = {i: [] for i in range(num_data)}
        zs_history_dict = {i: [] for i in range(num_data)}
        lambda_history_dict = {i: [] for i in range(num_data)}

        for _ in range(max_iter):
            aggregator.run_step()
            for i in range(num_data):
                worker = aggregator.admm_workers[i]
                xs_history_dict[i].append(worker.xs[0].item())
                zs_history_dict[i].append(get_value(worker.zs))
                lambda_history_dict[i].append(worker.lambdas[0].item())

        n_rows = 3
        n_cols = num_data

        xs = list(range(max_iter))

        for i in range(num_data):
            plt.subplot(n_rows, n_cols, i + 1)
            # plot zs
            plt.hlines(get_value(mean), 0, max_iter-1)
            plt.plot(xs, zs_history_dict[i])
            if i == 0:
                plt.ylabel('z')

            plt.subplot(n_rows, n_cols, n_cols + i + 1)
            # plot xs
            w = get_value(models[i].state_dict())
            plt.hlines(w, 0, max_iter-1)
            plt.plot(xs, xs_history_dict[i])
            if i == 0:
                plt.ylabel('x')

            plt.subplot(n_rows, n_cols, 2 * n_cols + i + 1)
            # plot lambdas
            plt.plot(xs, lambda_history_dict[i])
            if i == 0:
                plt.ylabel('lambda')

        plt.show()


if __name__ == "__main__":
    main()
