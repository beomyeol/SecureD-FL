from __future__ import absolute_import, division, print_function

import copy
import torch
import numpy as np
import random
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


def to_math_text(name):
    if name == 'lr':
        return r'$\rho$'
    else:
        return name


def run_admm_and_plot(aggregator_base, attr_name, attr_values, max_iter, mean,
                      plot_xs=False, plot_lambdas=False):
    num_workers = len(aggregator_base.admm_workers)
    n_rows = 1
    if plot_xs:
        xs_row = n_rows
        n_rows += 1
    if plot_lambdas:
        lambda_row = n_rows
        n_rows += 1
    n_cols = num_workers
    xs = list(range(1, max_iter+1))

    fig = plt.figure(figsize=(num_workers * 4, 4))
    gs = fig.add_gridspec(n_rows, n_cols)

    zs_axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, num_workers):
        zs_axes.append(fig.add_subplot(gs[0, i], sharey=zs_axes[0]))

    if plot_xs:
        xs_axes = [fig.add_subplot(gs[xs_row, i]) for i in range(num_workers)]
    if plot_lambdas:
        lambdas_axes = [fig.add_subplot(gs[lambda_row, i])
                        for i in range(num_workers)]

    for attr_value, color in zip(attr_values, COLORS[:len(attr_values)]):
        aggregator = copy.deepcopy(aggregator_base)
        setattr(aggregator, attr_name, attr_value)

        zs_history_list = [[] for i in range(num_workers)]
        xs_history_list = [[] for i in range(num_workers)]
        lambda_history_list = [[] for i in range(num_workers)]

        for _ in range(max_iter):
            for i in range(num_workers):
                worker = aggregator.admm_workers[i]
                lambda_history_list[i].append(worker.lambdas[0].item())

            aggregator.run_step()
            for i in range(num_workers):
                worker = aggregator.admm_workers[i]
                xs_history_list[i].append(worker.xs[0].item())

        for i in range(num_workers):
            worker = aggregator.admm_workers[i]
            zs_history_list[i] = [get_value(zs) for zs in worker.zs_history]

        aggregated_zs_history = [get_value(zs) for zs in aggregator.zs_history]
        print('{}={}, zs='.format(attr_name, attr_value))
        for i, zs in enumerate(zs_history_list):
            print('\t{}: {}'.format(i, zs))
        print('\tAVG: {}'.format(aggregated_zs_history))

        for i in range(num_workers):
            ax = zs_axes[i]
            # plot zs
            ax.plot(xs, zs_history_list[i],
                    color=color, label='{}={}'.format(
                to_math_text(attr_name), attr_value))
            ax.plot(xs, aggregated_zs_history, color=color, linestyle='dashed')
            ax.hlines(mean, 1, max_iter, linestyles='dotted')
            if i == 0:
                ax.set_ylabel('$z$')
            #ax.set_xticks(np.arange(1, max_iter+1, 1))
            ax.legend()

            if plot_xs:
                ax = xs_axes[i]
                w = get_value(aggregator.admm_workers[i].model.state_dict())
                ax.hlines(w, 1, max_iter)
                ax.hlines(mean, 1, max_iter, linestyles='dashed')
                ax.plot(xs, xs_history_list[i],
                        color=color, label='{}={}'.format(
                    to_math_text(attr_name), attr_value))
                if i == 0:
                    ax.set_ylabel('$x$')
                ax.set_xticks(np.arange(1, max_iter+1, 1))
                ax.legend()

            if plot_lambdas:
                ax = lambdas_axes[i]
                ax.plot(xs, lambda_history_list[i],
                        color=color, label='{}={}'.format(
                    to_math_text(attr_name), attr_value))
                if i == 0:
                    ax.set_ylabel(r'$\lambda$')
                ax.set_xticks(np.arange(1, max_iter+1, 1))
                ax.legend()

    plt.show()


def main():
    num_workers = 5
    device = torch.device('cpu')

    weights = [1/num_workers] * num_workers
    max_iter = 10

    models = generate_models(num_workers, device)

    def rho_gen_fn(lr):
        return random.uniform(0.9 * lr, 1.1 * lr)

    workers = [ADMMWorker(model, device, record_zs_history=True)
               for model in models]
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
    lrs = [3, 1, 0.3]
    run_admm_and_plot(aggregator_base, 'lr', lrs, max_iter, mean,
                      plot_xs=False, plot_lambdas=False)


if __name__ == "__main__":
    main()
