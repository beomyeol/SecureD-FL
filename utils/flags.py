from __future__ import absolute_import, division, print_function

DEFAULT_ARGS = {
    'epochs': 10,
    'local_epochs': 10,
    'batch_size': 32,
    'optimizer': 'rmsprop',
    'lr': 0.01,
    'log_every_n_steps': 10,
    'seed': 1234,
}


def add_dataset_flags(parser):
    parser.add_argument(
        '--dataset_dir', required=True, help='dataset root dir')
    parser.add_argument(
        '--dataset_download', action='store_true',
        help='download the dataset if not exists')
    parser.add_argument(
        '--max_num_clients', type=int,
        help='max number of clients to use in the dataset')
    parser.add_argument(
        '--split_ratios', help='comma seperated split ratios')
    parser.add_argument(
        '--max_num_users_per_worker', type=int,
        help='max number of users that each worker can hold')


def add_base_flags(parser):
    parser.add_argument(
        '--num_workers', type=int, required=True,
        help='number of workers to use in simulation')
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
        '--drop_last', action='store_true', help='drop last non-full batch')
    parser.add_argument(
        '--optimizer', default=DEFAULT_ARGS['optimizer'],
        help='optimizer (default={})'.format(DEFAULT_ARGS['optimizer']))
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
        '--validation_period', type=int,
        help='run validation in every given epochs')
    add_dataset_flags(parser)


def add_admm_flags(parser):
    parser.add_argument(
        '--use_admm', action='store_true',
        help='use ADMM-based average for aggregation')
    parser.add_argument(
        '--admm_threshold', type=float,
        help='threshold to determine early stopping for ADMM average')
    parser.add_argument(
        '--admm_max_iter', type=int, help='max iteration for ADMM average')
    parser.add_argument(
        '--admm_lr', type=float, help='learning rate for ADMM')
    parser.add_argument(
        '--admm_decay_period', type=int, help='ADMM learning rate decay period')
    parser.add_argument(
        '--admm_decay_rate', type=float, help='ADMM learning rate decay rate')


def check_admm_args(args):
    if args.use_admm:
        assert args.admm_max_iter
        assert args.admm_lr

        if args.admm_decay_period:
            assert args.admm_decay_period > 0
            assert args.admm_decay_rate
