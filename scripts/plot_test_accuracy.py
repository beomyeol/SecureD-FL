from __future__ import division, print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np

from parse_test_accuracy import get_test_accuracies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', help='comma-seperated input log file paths')
    parser.add_argument('--labels', help='comma-seperated labels for each log')

    args = parser.parse_args()

    inputs = args.inputs.split(',')
    labels = args.labels.split(',')

    for input_path, label in zip(inputs, labels):
        with open(input_path, 'r') as f:
            _, num_corrects_dict, num_totals_dict = get_test_accuracies(f)

        epochs = sorted(list(num_corrects_dict.keys()))
        num_corrects_list = np.array([num_corrects_dict[epoch]
                                      for epoch in epochs])
        num_totals_list = np.array([num_totals_dict[epoch] for epoch in epochs])
        plt.plot(epochs, num_corrects_list/num_totals_list, '-o', label=label)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
