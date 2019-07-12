import argparse
import time
import numpy as np


FORMAT = '%Y-%m-%d %H:%M:%S,%f'


def get_value(line, name):
    idx = line.find(name)
    if idx == -1:
        return None

    start_idx = idx + len(name) + 1
    end_idx = line.find(',', idx)

    value = line[start_idx:end_idx].strip()
    if value.startswith('['):
        value = value[1:-1].split('/')
    return value


def get_timestamp(line):
    idx = line.find(' - ')
    if idx == -1:
        return None
    return time.mktime(time.strptime(line[:idx].strip(), FORMAT))


def put_value(dic, key, value):
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]


def main(path):
    comp_times_dict = {}
    comm_times_dict = {}
    batches_dict = {}

    start_ts = None

    with open(path, 'r') as f:
        for line in f:
            ts = get_timestamp(line)
            if start_ts is None:
                start_ts = ts
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
            if epoch and int(epoch[0]) == 0:
                batches = get_value(line, 'batches')
                if batches:
                    batches_dict[rank] = int(batches[1])

    print('Comm times:')
    for rank, values in sorted(comm_times_dict.items()):
        print('\trank={}, mean={}, stdev={}'.format(
            rank, np.mean(values), np.std(values)))

    print('Comp times:')
    for rank, values in sorted(comp_times_dict.items()):
        mean = np.mean(values)
        num_batches = batches_dict[rank]
        print('\trank={}, mean={}, stdev={}, #batches={}, time_per_batch={}'.format(
            rank, mean, np.std(values), num_batches, mean/num_batches))

    print('Total time: {}'.format(ts - start_ts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1, help='path')

    args = parser.parse_args()

    main(args.input[0])
