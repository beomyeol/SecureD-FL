"""Tune ADMM hyperparameters."""
from __future__ import absolute_import, division, print_function

import argparse

import numpy as np
import torch

import utils.mock as mock
import utils.ops as ops
from utils.admm_parameter_tuner import ADMMParameterTuner


def calculate_distance_z_and_param(admm_workers):
    diffs = []
    for admm_worker in admm_workers:
        diffs.append(ops.calculate_distance(admm_worker.model.state_dict(),
                                            admm_worker.zs))
    retval = np.mean(diffs)
    print('Avg distance between the parameters and zs: ', str(retval))
    return retval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help='checkpoint path')
    parser.add_argument('--early_stop', type=float,
                        help='Use early stop with the provided threshold')
    parser.add_argument('--max_iters', type=int, required=True,
                        help='max iters')
    parser.add_argument('--gpu_id', type=int, help='gpu id')

    args = parser.parse_args()

    if args.gpu_id is not None:
        device_str = 'cuda:%d' % args.gpu_id
    else:
        device_str = 'cpu'

    print(device_str)

    device = torch.device(device_str)

    save_dict = torch.load(args.INPUT, map_location=device_str)
    models = [mock.MockModel(state_dict, device)
              for state_dict in save_dict.values()]

    lrs = [
        #  300, 150, 100,
        70, 50, 30, 10,
        7, 5, 3, 1,
        7e-1, 5e-1, 3e-1, 1e-1,
        7e-2, 5e-2, 3e-2, 1e-2,
        7e-3, 5e-3, 3e-3, 1e-3,
        7e-4, 5e-4, 3e-4, 1e-4,
    ]
    decay_rates = [
        1,
        0.7, 0.5, 0.3, 0.1,
        7e-2, 5e-2, 3e-2, 1e-2,
        7e-3, 5e-3, 3e-3, 1e-3,
        7e-4, 5e-4, 3e-4, 1e-4,
        7e-5, 5e-5, 3e-5, 1e-5,
        7e-6, 5e-6, 3e-6, 1e-6,
    ]
    if args.max_iters == 1:
        decay_periods = [1]
    else:
        decay_periods = np.arange(1, args.max_iters)
    thresholds = [0]
    max_iters = [args.max_iters]

    tuner = ADMMParameterTuner(
        models=models,
        device=device,
        lrs=lrs,
        decay_rates=decay_rates,
        decay_periods=decay_periods,
        thresholds=thresholds,
        max_iters=max_iters,
        early_stop_threshold=args.early_stop,
    )
    tuner.run()

    result = tuner.get()[0]

    print('Best parameter result:')
    print('\titer:', result.iter)
    print('\tmse:', result.mse)
    print('\tparameters:', result.parameters)


if __name__ == "__main__":
    main()
