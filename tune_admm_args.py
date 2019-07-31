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

from sequential.worker import fedavg
import utils.ops as ops
import utils.mock as mock
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
    parser.add_argument('INPUT', nargs=1, help='checkpoint path')

    args = parser.parse_args()

    device = torch.device('cpu')

    save_dict = torch.load(args.INPUT[0], map_location='cpu')
    models = [mock.MockModel(state_dict, device)
              for state_dict in save_dict.values()]

    lrs = [1e-1, 7e-2, 5e-2, 3e-2, 1e-2, 7e-3, 5e-3, 3e-3, 1e-3, 7e-4]
    decay_rates = [1, 0.9, 0.8, 0.5]
    decay_periods = [1, 2, 4, 8]
    thresholds = [1e-6]
    max_iters = [10]

    tuner = ADMMParameterTuner(
        models=models,
        device=torch.device('cpu'),
        lrs=lrs,
        decay_rates=decay_rates,
        decay_periods=decay_periods,
        thresholds=thresholds,
        max_iters=max_iters,
    )
    tuner.run()

    result = tuner.get()[0]

    print('Best parameter result:')
    print('\titer:', result.iter)
    print('\tmse:', result.mse)
    print('\tparameters:', result.parameters)


if __name__ == "__main__":
    main()
