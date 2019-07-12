from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

from parse_utils import get_value, get_timestamp


def put_value(dic, key, value):
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]


def main(path):
    comp_times_dict = {}
    comm_times_dict = {}
    local_epochs_dict = {}
    batches_dict = {}
    elapsed_time_list = []

    start_ts = None

    with open(path, 'r') as f:
        for line in f:
            ts = get_timestamp(line)
            if start_ts is None:
                start_ts = ts
            elapsed_time = get_value(line, 'elapsed_time')
            if elapsed_time:
                elapsed_time_list.append(float(elapsed_time))

            rank = get_value(line, 'rank')
            if not rank:
                continue
            rank = int(rank)
            comp_time = get_value(line, 'comp_time')
            comm_time = get_value(line, 'comm_time')
            if comp_time:
                put_value(comp_times_dict, rank,
                          float(comp_time.split(' ')[0]))
            if comm_time:
                put_value(comm_times_dict, rank,
                          float(comm_time.split(' ')[0]))
            epoch = get_value(line, 'epoch')
            local_epoch = get_value(line, 'local_epoch')
            if (epoch and local_epoch and
                    int(epoch[0]) == 0 and int(local_epoch[0]) == 0):
                batches = get_value(line, 'batches')
                if batches:
                    batches_dict[rank] = int(batches[1])
                    local_epochs_dict[rank] = int(local_epoch[1])

    print('Comm times:')
    for rank, values in sorted(comm_times_dict.items()):
        print('\trank={}, mean={}, stdev={}'.format(
            rank, np.mean(values), np.std(values)))

    print('Comp times:')
    for rank, values in sorted(comp_times_dict.items()):
        mean = np.mean(values)
        num_batches = batches_dict[rank]
        local_epochs = local_epochs_dict[rank]
        num_batches_per_epoch = num_batches * local_epochs
        print('\trank=%d, mean=%.6f, stdev=%.6f, #batches=%d, '
              'local_epochs=%d, #batches_per_epoch=%d' %
              (rank, mean, np.std(values), num_batches,
               local_epochs, num_batches_per_epoch))

    print('Total time: {}'.format(ts - start_ts))
    if elapsed_time_list:
        print('Elapsed times: {}'.format(np.sum(elapsed_time_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1, help='log path')

    args = parser.parse_args()

    main(args.input[0])
