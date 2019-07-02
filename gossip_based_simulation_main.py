from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F

import datasets.femnist as femnist
from gossip.worker import Worker
from nets.lenet import LeNet
from utils.train import TrainArguments
import utils.flags as flags


DEFAULT_ARGS = {
    'port': 12345,
    'num_gossips': 1,
}


def run_worker(rank, cluster_spec, args):
    partition = femnist.get_partition(
        args.dataset_dir, rank, len(cluster_spec), args.seed,
        download=args.dataset_download,
        max_num_users=args.max_num_users)
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    validation = (None, None)
    if args.validation_period:
        test_partition = femnist.get_partition(
            args.dataset_dir, rank, len(cluster_spec), args.seed,
            train=False,
            download=args.dataset_download,
            max_num_users=args.max_num_users)
        assert partition.client_ids == test_partition.client_ids
        test_data_loader = torch.utils.data.DataLoader(
            test_partition, batch_size=args.batch_size)
        validation = (args.validation_period, test_data_loader)

    admm_kwargs = None
    if args.use_admm:
        admm_kwargs = {
            'max_iter': args.admm_max_iter,
            'tolerance': args.admm_tolerance,
            'lr': args.admm_lr
        }

    device = torch.device('cpu')
    model = LeNet()
    train_args = TrainArguments(
        data_loader=data_loader,
        device=device,
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
    )

    worker = Worker(rank, cluster_spec, args.num_gossips,
                    args.seed, admm_kwargs)
    worker.run(args.epochs, args.local_epochs, train_args, validation)


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
