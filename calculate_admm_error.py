from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import numpy as np
import torch

import utils.mock as mock
from utils.admm_parameter_tuner import ADMMParameterTuner

HPARAMS = {
    1: {
        'lrs': [3],
        'decay_rates': [1],
        'decay_periods': [1],
    },
    2: {
        'lrs': [3],
        'decay_rates': [0.0003],
        'decay_periods': [1],
    },
    #  3: {
        #  'lrs': [10],
        #  'decay_rates': [0.008],
        #  'decay_periods': [1],
    #  },
    3: {
        'lrs': [7],
        'decay_rates': [0.01],
        'decay_periods': [1],
    },
    #  4: {
        #  'lrs': [7],
        #  'decay_rates': [0.0003],
        #  'decay_periods': [2],
    #  },
    4: {
        'lrs': [70],
        'decay_rates': [7e-6],
        'decay_periods': [2],
    },
    #  4: {
        #  'lrs': [300],
        #  'decay_rates': [1e-6],
        #  'decay_periods': [2],
    #  },
    5: {
        'lrs': [50],
        'decay_rates': [7e-06],
        'decay_periods': [3],
    },
    6: {
        'lrs': [100],
        'decay_rates': [7e-6],
        'decay_periods': [3],
    },
    7: {
        'lrs': [100],
        'decay_rates': [7e-6],
        'decay_periods': [4],
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', help='checkpoint dir path')
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

    hparams = HPARAMS[args.max_iters]
    print(hparams)

    mses = []

    for ckpt_path in glob.glob(os.path.join(args.INPUT, '*.ckpt')):
        print(ckpt_path)

        save_dict = torch.load(ckpt_path, map_location=device_str)
        models = [mock.MockModel(state_dict, device)
                  for state_dict in save_dict.values()]

        tuner = ADMMParameterTuner(
            models=models,
            device=device,
            thresholds=[0],
            max_iters=[args.max_iters],
            early_stop_threshold=0.0,
            **hparams)
        tuner.run()

        result = tuner.get()[0]
        mses.append(result.mse)

    print('MSEs:')
    for mse in mses:
        print('\t{}'.format(mse))
    #  print('Average MSE:')
    #  print('\t{}'.format(np.mean(mses)))
    #  print('STDEV:')
    #  print('\t{}'.format(np.std(mses)))


if __name__ == "__main__":
    main()
