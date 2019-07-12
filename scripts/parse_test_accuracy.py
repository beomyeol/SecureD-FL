from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

from parse_utils import get_value


def main(args):
    log_path = args.input[0]
    num_workers = args.num_workers

    with open(log_path, 'r') as f:
        test_accuracy_list = []
        correct_list = []
        total_list = []
        current_epoch = None

        for line in f:
            test_accuracy = get_value(line, 'test accuracy')
            if not test_accuracy:
                continue

            epoch= int(get_value(line, 'epoch')[0])
            if current_epoch:
                assert epoch == current_epoch
            else:
                current_epoch = epoch

            test_accuracy, remains = test_accuracy.split('[')
            correct, remains = remains.split('/')
            total = remains[:-1]

            test_accuracy_list.append(float(test_accuracy))
            correct_list.append(int(correct))
            total_list.append(int(total))

            if len(test_accuracy_list) == num_workers:
                print('epoch: %d' % current_epoch)
                for test_accuracy in test_accuracy_list:
                    print(test_accuracy)

                print(np.sum(correct_list) / np.sum(total_list))

                test_accuracy_list = []
                correct_list = []
                total_list = []
                current_epoch = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs=1, help='log path')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='#workers used in the simulation')

    args = parser.parse_args()
    main(args)
