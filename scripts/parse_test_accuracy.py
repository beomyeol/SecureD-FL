from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

from parse_utils import get_value


def get_test_accuracies(f):
    acc_dict = {}  # epoch# -> (rank -> test accuracy)
    num_corrects_dict = {}  # epoch# -> (rank -> # corrected prediction)
    num_totals_dict = {}  # rank -> # total prediction

    def put(d, epoch, rank, value):
        if epoch not in d:
            d[epoch] = {rank: [value]}
            return

        if rank in d[epoch]:
            d[epoch][rank].append(value)
        else:
            d[epoch][rank] = [value]

    for line in f:
        test_accuracy = get_value(line, 'test accuracy')
        if not test_accuracy:
            continue

        epoch = int(get_value(line, 'epoch')[0])
        rank = int(get_value(line, 'rank'))

        test_accuracy, remains = test_accuracy.split('[')
        correct, remains = remains.split('/')
        correct = int(correct)
        total = int(remains[:-1])

        put(acc_dict, epoch, rank, float(test_accuracy))
        put(num_corrects_dict, epoch, rank, correct)
        if rank in num_totals_dict:
            assert num_totals_dict[rank] == total
        else:
            num_totals_dict[rank] = total

    return acc_dict, num_corrects_dict, num_totals_dict


def main(args):
    log_path = args.input

    with open(log_path, 'r') as f:
        retval = get_test_accuracies(f)
        test_accuracy_dict, num_corrects_dict, num_totals_dict = retval

    total_acc_list = []
    for epoch in sorted(test_accuracy_dict):
        print('epoch: %d' % epoch)

        test_accuracy_per_site = test_accuracy_dict[epoch]
        after_aggr_test_available = False
        for rank, site_test_acc in sorted(test_accuracy_per_site.items()):
            print('\t{}: {}'.format(rank, site_test_acc[0]))
            after_aggr_test_available = len(site_test_acc) > 1

        num_corrects_per_site = num_corrects_dict[epoch]
        assert len(num_corrects_per_site) == len(test_accuracy_per_site)
        assert len(num_corrects_per_site) == len(num_totals_dict)
        num_corrects_sum = sum(
            [num_corrects[0]
             for num_corrects in num_corrects_per_site.values()])
        num_totals_sum = sum(num_totals_dict.values())

        total_acc = num_corrects_sum/num_totals_sum
        print('\tTotal: {} [{}/{}]'.format(
            total_acc, num_corrects_sum, num_totals_sum))
        total_acc_item = [total_acc]

        if after_aggr_test_available:
            print('epoch: %d (after aggregation)' % epoch)
            for rank, site_test_acc in sorted(test_accuracy_per_site.items()):
                print('\t{}: {}'.format(rank, site_test_acc[1]))

            num_corrects_sum = sum(
                [num_corrects[1]
                 for num_corrects in num_corrects_per_site.values()])
            total_acc = num_corrects_sum/num_totals_sum
            print('\tTotal: {} [{}/{}]'.format(
                total_acc, num_corrects_sum, num_totals_sum))
            total_acc_item.append(total_acc)

        total_acc_list.append(total_acc_item)

    # print best test accuracy across the whole execution
    test_acc_tensor = np.array(
        [[site_test_acc for rank, site_test_acc in test_acc_per_site.items()]
         for epoch, test_acc_per_site in sorted(test_accuracy_dict.items())])

    best_test_accs = test_acc_tensor.max(axis=0)
    best_total_accs = np.array(total_acc_list).max(axis=0)

    print('Best:')
    for i, test_acc in enumerate(best_test_accs):
        print('\t{}: {}'.format(i, test_acc[0]))
    print('\tTotal: {}'.format(best_total_accs[0]))

    if best_test_accs.shape[-1] == 2:
        print('Best after aggregation:')
        for i, test_acc in enumerate(best_test_accs):
            print('\t{}: {}'.format(i, test_acc[1]))
        print('\tTotal: {}'.format(best_total_accs[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='log path')

    args = parser.parse_args()
    main(args)
