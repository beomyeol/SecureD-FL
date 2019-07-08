from __future__ import absolute_import, division, print_function

import argparse
import functools
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
                                      **partition_kwargs,
                                      **vars(args))
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    test_args = None
    if args.validation_period:
        test_partition = dataset.get_partition(rank=rank,
                                               world_size=args.num_workers,
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
                    args.init_method, admm_kwargs=admm_kwargs)
    worker.run(args.epochs, args.local_epochs, train_args, test_args,
               without_sync=args.wo_sync)


DEFAULT_ARGS = {
    'init_method': 'tcp://127.0.0.1:23456',
    'model': 'lenet',
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
