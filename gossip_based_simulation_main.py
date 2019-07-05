from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F

import datasets.femnist as femnist
from gossip.worker import Worker
from nets.lenet import LeNet
from utils.train import TrainArguments, train_model
from utils.test import TestArguments
import utils.flags as flags


DEFAULT_ARGS = {
    'port': 12345,
    'num_gossips': 1,
}


def run_worker(rank, cluster_spec, args):
    device = torch.device('cpu')
    model = LeNet()

    partition = femnist.get_partition(rank=rank,
                                      world_size=len(cluster_spec),
                                      **vars(args))
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    test_args = None
    if args.validation_period:
        test_partition = femnist.get_partition(rank=rank,
                                               world_size=len(cluster_spec),
                                               train=False,
                                               **vars(args))
        assert partition.client_ids == test_partition.client_ids
        test_data_loader = torch.utils.data.DataLoader(
            test_partition, batch_size=args.batch_size)
        test_args = TestArguments(
            data_loader=test_data_loader,
            model=model,
            device=device,
            period=args.validation_period)

    train_args = TrainArguments(
        data_loader=data_loader,
        device=device,
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
        train_fn=train_model,
    )

    worker = Worker(rank, cluster_spec, args.num_gossips, args.seed)
    worker.run(args.epochs, args.local_epochs, train_args, test_args)


def main():
    parser = argparse.ArgumentParser()
    flags.add_base_flags(parser)
    parser.add_argument(
        '--port', type=int, default=DEFAULT_ARGS['port'],
        help='base port number (default={})'.format(DEFAULT_ARGS['port']))
    parser.add_argument(
        '--num_gossips', type=int, default=DEFAULT_ARGS['num_gossips'],
        help='number of gossips per each epoch (default={})'.format(
            DEFAULT_ARGS['num_gossips']))

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cluster_spec = ['localhost:%d' % (args.port + i)
                    for i in range(args.num_workers)]

    processes = []
    for rank in range(args.num_workers):
        p = mp.Process(target=run_worker,
                       args=(rank, cluster_spec, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
