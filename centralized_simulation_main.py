from __future__ import absolute_import, division, print_function

import argparse
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
import time
from torchvision import transforms

from nets.lenet import LeNet
from centralized.master import Master
from centralized.worker import Worker
import datasets.femnist as femnist
from utils.train import TrainArguments, train_model
import utils.flags as flags

DEFAULT_ARGS = {
    'init_method': 'tcp://127.0.0.1:23456',
}


def run_master(device, model, args):
    # TODO: support multiple datasets
    test_dataset = femnist.FEMNISTDataset(args.dataset_dir, train=False,
                                          transform=transforms.ToTensor(),
                                          only_digits=True)
    if args.validation_period:
        client_ids = list(test_dataset.client_ids())
        if args.max_num_users_per_worker:
            rng = random.Random()
            rng.seed(args.seed)
            rng.shuffle(client_ids)
            del client_ids[args.max_num_users_per_worker:]

        test_datasets = [test_dataset.get_client_dataset(client_id)
                         for client_id in client_ids]

        test_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset(test_datasets),
            batch_size=args.batch_size)

        test_args = test_data_loader, args.validation_period
    else:
        test_args = None

    master = Master(model, device, args.num_workers, args.init_method,
                    sample_size=args.sample_size, seed=args.seed)
    master.run(args.epochs, test_args)


def run_worker(rank, device, model, args):
    # TODO: support multiple datasets
    dataset = femnist.get_partition(rank=rank-1,
                                    world_size=args.num_workers,
                                    **vars(args))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    worker = Worker(model, device, rank, args.num_workers, args.init_method,
                    sample_size=args.sample_size, seed=args.seed)

    train_args = TrainArguments(
        data_loader=data_loader,
        device=torch.device('cpu'),
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=args.lr),
        loss_fn=F.nll_loss,
        log_every_n_steps=args.log_every_n_steps,
        train_fn=train_model,
    )

    worker.run(args.epochs, args.local_epochs, train_args)


def main():
    parser = argparse.ArgumentParser()
    flags.add_base_flags(parser)
    parser.add_argument(
        '--sample_size', type=int,
        help='size of worker samples in each epoch. '
             'If not set, all workers paricipate in each epoch.')
    parser.add_argument(
        '--init_method', default=DEFAULT_ARGS['init_method'],
        help='init method to use for torch.distributed (default={})'.format(
            DEFAULT_ARGS['init_method']))

    args = parser.parse_args()

    # TODO: support GPU
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

    model = LeNet()

    master_process = mp.Process(target=run_master, args=(device, model, args))
    master_process.start()
    time.sleep(0.1)

    worker_processes = []
    for rank in range(1, args.num_workers + 1):
        p = mp.Process(target=run_worker,
                       args=(rank, device, model, args))
        p.start()
        worker_processes.append(p)

    master_process.join()
    for p in worker_processes:
        if p.is_alive():
            p.terminate()
        else:
            p.join()


if __name__ == "__main__":
    main()
