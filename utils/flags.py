from __future__ import absolute_import, division, print_function

import argparse


DEFAULT_ARGS = {
    'epochs': 10,
    'local_epochs': 10,
    'batch_size': 32,
    'lr': 0.001,
    'log_every_n_steps': 10,
    'seed': 1234,
}

ADMM_DEFAULT_ARGS = {
    'admm_tolerance': 0.01,
    'admm_max_iter': 20,
    'admm_lr': 0.01,
}


def add_base_flags(parser):
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers to use in simulation')
    parser.add_argument(
        '--dataset_dir', required=True, help='dataset root dir')
    parser.add_argument(
        '--dataset_download', action='store_true',
        help='download the dataset if not exists')
    parser.add_argument(
        '--epochs', type=int, default=DEFAULT_ARGS['epochs'],
        help='number of epochs to train (default={})'.format(
            DEFAULT_ARGS['epochs']))
    parser.add_argument(
        '--local_epochs', type=int, default=DEFAULT_ARGS['local_epochs'],
        help='number of local epochs in each global epoch (default={})'.format(
            DEFAULT_ARGS['local_epochs']))
    parser.add_argument(
        '--batch_size', type=int, default=DEFAULT_ARGS['batch_size'],
        help='batch size (default={})'.format(DEFAULT_ARGS['batch_size']))
    parser.add_argument(
        '--lr', type=float, default=DEFAULT_ARGS['lr'],
        help='learning rate (default={})'.format(DEFAULT_ARGS['lr']))
    parser.add_argument(
        '--log_every_n_steps', type=int,
        default=DEFAULT_ARGS['log_every_n_steps'],
        help='log every n steps (default={})'.format(
            DEFAULT_ARGS['log_every_n_steps']))
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_ARGS['seed'],
        help='random seed (default={})'.format(DEFAULT_ARGS['seed']))
    parser.add_argument(
        '--max_num_users', type=int,
        help='max number of users that each worker can hold')
    parser.add_argument(
        '--validation_period', type=int,
        help='run validation in every given epochs')


def add_admm_flags(parser):
    parser.add_argument(
        '--use_admm', action='store_true',
        help='Use ADMM-based average for aggregation')
    parser.add_argument(
        '--admm_tolerance', type=float,
        default=ADMM_DEFAULT_ARGS['admm_tolerance'],
        help='Tolerance for ADMM average (default={})'.format(
            ADMM_DEFAULT_ARGS['admm_tolerance']))
    parser.add_argument(
        '--admm_max_iter', type=int,
        default=ADMM_DEFAULT_ARGS['admm_max_iter'],
        help='max iteration for admm average (default={})'.format(
            ADMM_DEFAULT_ARGS['admm_max_iter']))
    parser.add_argument(
        '--admm_lr', type=float,
        default=ADMM_DEFAULT_ARGS['admm_lr'],
        help='learning rate for ADMM (default={})'.format(
            ADMM_DEFAULT_ARGS['admm_lr']))