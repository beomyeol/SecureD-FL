from __future__ import absolute_import, division, print_function

import argparse
import torch
import math
import time
from torchvision import transforms

from torch.utils.model_zoo import tqdm

from datasets.dataset_factory import get_load_dataset_fn
from datasets.partition import get_partition
import utils.logger as logger
import utils.flags as flags

_LOGGER = logger.get_logger(__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True,
                        help='dataset name to calculate its sensitivity')
    parser.add_argument('--id', type=int, required=True,
                        help='process id')
    parser.add_argument('--num', type=int, required=True,
                        help='total #processes')
    parser.add_argument('--gpu_id', type=int, help='gpu id')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed for partitioning')
    flags.add_dataset_flags(parser)

    args = parser.parse_args()

    if args.gpu_id:
        device = torch.device('cuda:%d' % args.gpu_id)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    dataset_fn = get_load_dataset_fn(args.name)

    dataset = dataset_fn(train=True, transform=torch.flatten, **vars(args))
    _LOGGER.info('#clients in the dataset to use: %d', len(dataset.client_ids))

    batch_size = args.batch_size

    # A single partition
    partition = get_partition(dataset, rank=0, world_size=1, seed=args.seed)
    _LOGGER.info('id=%d, dataset #users: %d, partition #users: %d',
                 args.id,
                 len(dataset.client_ids),
                 len(partition.client_ids))

    num_items = int(len(partition) / args.num)

    start = num_items * args.id
    end = num_items * (args.id + 1)
    _LOGGER.info('id=%d, range=[%d:%d]', args.id, start, end)

    if args.id == args.num - 1:
        end = len(partition)

    max_l1_loss = None

    for i in range(start, end):
        start_ts = time.time()
        data = partition[i][0].to(device)
        subset = torch.utils.data.Subset(
            partition, list(range(i + 1, len(partition))))
        data_loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=False)

        for others, _ in data_loader:
            others = others.to(device)
            # TODO: fix this to calculate the correct sensitivity
            loss = torch.abs(others - data).mean(dim=1)
            l1_loss = loss.max().item()

            if max_l1_loss is None or l1_loss > max_l1_loss:
                max_l1_loss = l1_loss

        _LOGGER.info('id=%d [%d/%d] max_loss=%f, elapsed_time=%f',
                     args.id, i - start, end - start, max_l1_loss,
                     time.time() - start_ts)


    print('Sensitivity:', max_l1_loss)


if __name__ == "__main__":
    main()
