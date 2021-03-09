from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import os


from parse_test_accuracy import get_test_accuracies


def add_to_dict(dic, key, value):
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='log dir path')

    args = parser.parse_args()

    aggregated_acc_dict = {}  # key: epoch, value: a list of test accuracies
    aggregated_num_corrects_dict = {}  # key: epoch, value: a list of num_corrects
    aggregated_num_totals_dict = {}  # key: epoch, value: a list of num_totals

    for trial_dir in os.listdir(args.dir):
        log_path = os.path.join(args.dir, trial_dir, 'run.log')
        with open(log_path, 'r') as f:
            _, num_corrects_dict, num_totals_dict = get_test_accuracies(f)

        epochs = sorted(list(num_corrects_dict.keys()))
        num_corrects_list = np.array([num_corrects_dict[epoch]
                                      for epoch in epochs])
        num_totals_list = np.array([num_totals_dict[epoch]
                                    for epoch in epochs])
        test_accuracy_list = num_corrects_list / num_totals_list

        for epoch, acc, c, t in zip(
                epochs, test_accuracy_list, num_corrects_list, num_totals_list):
            add_to_dict(aggregated_acc_dict, epoch, acc)
            add_to_dict(aggregated_num_corrects_dict, epoch, c)
            add_to_dict(aggregated_num_totals_dict, epoch, t)

    epochs = sorted(list(aggregated_acc_dict.keys()))
    for epoch in epochs:
        log = 'epoch: [{}/{}], test accuracy: {}[{}/{}]'.format(
            epoch, len(epochs), np.mean(aggregated_acc_dict[epoch]),
            np.mean(aggregated_num_corrects_dict[epoch]),
            np.mean(aggregated_num_totals_dict[epoch]))
        print(log)


if __name__ == "__main__":
    main()
