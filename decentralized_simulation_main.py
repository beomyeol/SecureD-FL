from __future__ import absolute_import, division, print_function

import argparse
import functools
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

import datasets.femnist as femnist
import datasets.shakespeare as shakespeare
from nets.lenet import LeNet
from nets.rnn import RNN
from decentralized.worker import Worker
from utils.train import TrainArguments, train_model, train_rnn
from utils.test import TestArguments
import utils.flags as flags
import utils.logger as logger


_LOGGER = logger.get_logger(__file__)


def run_worker(rank, args):
    model_kwargs = {}
    if args.model == 'lenet':
        model = LeNet()
        dataset = femnist
        partition_kwargs = {}
        train_fn = train_model
    elif args.model == 'rnn':
        dataset = shakespeare
        model = RNN(
            vocab_size=len(shakespeare.ShakespeareDataset._VOCAB),
            embedding_dim=100,
            hidden_size=128)
        partition_kwargs = {'seq_length': 50}
        hidden = model.init_hidden(args.batch_size)
        train_fn = functools.partial(train_rnn, hidden=hidden)
    else:
        raise ValueError('Unknown model: ' + args.model)

    device = torch.device('cpu')
    partition = dataset.get_partition(rank=rank,
                                      world_size=args.num_workers,
                                      ratios=args.split_ratios,
                                      **partition_kwargs,
                                      **vars(args))
    _LOGGER.info('rank=%d, #clients=%d', rank, len(partition.client_ids))
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    test_args = None
    if args.validation_period:
        test_partition = dataset.get_partition(rank=rank,
                                               world_size=args.num_workers,
                                               ratios=args.split_ratios,
                                               train=False,
                                               **partition_kwargs,
                                               **vars(args))
        assert partition.client_ids == test_partition.client_ids
        test_data_loader = torch.utils.data.DataLoader(
            test_partition, batch_size=args.batch_size)
        test_args = TestArguments(
            data_loader=test_data_loader,
            model=model,
            device=device,
            period=args.validation_period)

    admm_kwargs = None
    if args.use_admm:
        admm_kwargs = {
            'max_iter': args.admm_max_iter,
            'tolerance': args.admm_tolerance,
            'lr': args.admm_lr
        }

    train_args = TrainArguments(
        data_loader=data_loader,
        device=device,
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
        train_fn=train_fn,
    )

    worker = Worker(rank, args.num_workers,
                    args.init_method, args.timeout,
                    admm_kwargs=admm_kwargs)

    local_epochs = args.local_epochs
    if args.adjust_local_epochs:
        num_batches = []
        for i in range(args.num_workers):
            num_batches.append(
                len(dataset.get_partition(rank=i,
                                          world_size=args.num_workers,
                                          **partition_kwargs,
                                          **vars(args))))
        lcm = np.lcm.reduce(num_batches)
        ratio = lcm / num_batches
        ratio *= args.local_epochs * args.num_workers / np.sum(ratio)
        local_epochs = int(ratio[rank])

    worker.run(args.epochs, local_epochs, train_args, test_args,
               without_sync=args.wo_sync)


DEFAULT_ARGS = {
    'init_method': 'tcp://127.0.0.1:23456',
    'model': 'lenet',
    'timeout': 1800,
}


def main():
    parser = argparse.ArgumentParser()
    flags.add_base_flags(parser)
    flags.add_admm_flags(parser)
    parser.add_argument(
        '--init_method', default=DEFAULT_ARGS['init_method'],
        help='init method to use for torch.distributed (default={})'.format(
            DEFAULT_ARGS['init_method']))
    parser.add_argument(
        '--model', default=DEFAULT_ARGS['model'],
        help='name of ML model to train (default={})'.format(
            DEFAULT_ARGS['model']))
    parser.add_argument(
        '--wo_sync', action='store_true', help='disable the synchronization')
    parser.add_argument(
        '--timeout', type=int, default=DEFAULT_ARGS['timeout'],
        help='timeout for torch.dist in sec (default={})'.format(
            DEFAULT_ARGS['timeout']))
    parser.add_argument(
        '--adjust_local_epochs', action='store_true',
        help='adjust local epochs depending on # mini-batches')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(target=run_worker,
                       args=(rank, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
