from __future__ import absolute_import, division, print_function

import argparse
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from datasets.femnist import FEMNISTDataset, FEMNISTDatasetPartitioner
from nets.lenet import LeNet
from decentralized.worker import Worker
from utils.train import TrainArguments
import utils.flags as flags


def run_worker(rank, args):
    dataset = FEMNISTDataset(args.dataset_dir, download=args.dataset_download,
                             only_digits=True, transform=transforms.ToTensor())
    partitioner = FEMNISTDatasetPartitioner(
        dataset,  args.num_workers, args.seed, args.max_num_users)
    partition = partitioner.get(rank)
    data_loader = torch.utils.data.DataLoader(
        partition, batch_size=args.batch_size, shuffle=True)

    validation = (None, None)
    if args.validation_period:
        test_dataset = FEMNISTDataset(args.dataset_dir, train=False,
                                      only_digits=True, transform=transforms.ToTensor())
        test_partitioner = FEMNISTDatasetPartitioner(
            test_dataset, args.num_workers, args.seed, args.max_num_users)
        test_partition = test_partitioner.get(rank)
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

    model = LeNet()
    train_args = TrainArguments(
        data_loader=data_loader,
        device=torch.device('cpu'),
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
    )

    worker = Worker(rank, args.num_workers,
                    args.init_method, admm_kwargs=admm_kwargs)
    worker.run(args.epochs, args.local_epochs, train_args, validation)


DEFAULT_ARGS = {
    'init_method': 'tcp://127.0.0.1:23456',
}


def main():
    parser = argparse.ArgumentParser()
    flags.add_base_flags(parser)
    flags.add_admm_flags(parser)
    parser.add_argument(
        '--init_method', default=DEFAULT_ARGS['init_method'],
        help='init method to use for torch.distributed (default={})'.format(
            DEFAULT_ARGS['init_method']))

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
