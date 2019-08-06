from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

from parse_utils import get_value


def get_test_accuracies(f):
    acc_list_dict = {}
    num_corrects_dict = {}
    num_totals_dict = {}

    def flush(current_epoch, acc_list, num_corrects, num_totals):
        acc_list_dict[current_epoch] = acc_list
        num_corrects_dict[current_epoch] = num_corrects
        num_corrects = 0
        num_totals_dict[current_epoch] = num_totals
        num_totals = 0

    acc_list = []
    num_corrects = 0
    num_totals = 0
    current_epoch = None
    num_workers = None

    for line in f:
        test_accuracy = get_value(line, 'test accuracy')
        if not test_accuracy:
            continue

        epoch = int(get_value(line, 'epoch')[0])
        if current_epoch is None:
            current_epoch = epoch
        elif epoch != current_epoch:
            assert epoch == current_epoch + 1
            if num_workers is None:
                num_workers = len(acc_list)
                print('#workers:', num_workers)
            else:
                assert len(acc_list) == num_workers, 'len=%d, line=%s' % (
                    len(acc_list), line)

        if num_workers is not None and len(acc_list) == num_workers:
            flush(current_epoch, acc_list, num_corrects, num_totals)
            acc_list = []
            current_epoch = epoch

        test_accuracy, remains = test_accuracy.split('[')
        correct, remains = remains.split('/')
        total = remains[:-1]

        acc_list.append(float(test_accuracy))
        num_corrects += int(correct)
        num_totals += int(total)

    flush(current_epoch, acc_list, num_corrects, num_totals)

    return acc_list_dict, num_corrects_dict, num_totals_dict


def main(args):
    log_path = args.input

    with open(log_path, 'r') as f:
        retval = get_test_accuracies(f)
        test_accuracy_list_dict, num_corrects_dict, num_totals_dict = retval

    for epoch in test_accuracy_list_dict:
        print('epoch:', epoch)
        for i, test_acc in enumerate(test_accuracy_list_dict[epoch]):
            print('\t{}:{}'.format(i, test_acc))

        print('\tTotal:{}'.format(
            num_corrects_dict[epoch]/num_totals_dict[epoch]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='log path')

    args = parser.parse_args()
    main(args)
