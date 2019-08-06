from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt


def plot_test_accuracy_by_num_minibatches():

    fixed_mini_batches = [106880, 213760, 320640, 427520,
                          534400, 641280, 748160, 855040, 961920, 1068800]
    fixed_accuracy = [0.986554663, 0.992554859, 0.99424471, 0.993852861,
                      0.994587578, 0.994489616, 0.994979428, 0.994612069, 0.994905956, 0.995175353]

    adjusted_mini_batches = [70673, 141346, 212019, 282692,
                             353365, 424038, 494711, 565384, 636057, 706730]
    adjusted_accuracy = [0.985403605, 0.991134404, 0.993779389, 0.994122257,
                         0.995126371, 0.994587578, 0.995101881, 0.994979428, 0.994734522, 0.995395768]

    plt.figure(figsize=(6, 4))
    plt.plot(fixed_mini_batches, fixed_accuracy, '-o', color='b', label='Fixed')
    plt.plot(adjusted_mini_batches, adjusted_accuracy,
             '-o', color='r', label='Adjusted')
    plt.xlabel('# processed mini-batches')
    plt.ylabel('Test accuracy')
    plt.legend(loc=4)
    plt.show()


def plot_comp_and_comm_time():
    comp_times = [276.87657, 181.642296, 91.4609575, 91.4505055,
                  45.436051, 44.782393, 44.5675445, 46.3714275, 46.167847, 45.4818375]
    comm_times = np.max(comp_times) - comp_times
    indices = np.arange(len(comp_times))
    width = 0.5

    plt.figure(figsize=(6, 3.5))
    comp = plt.bar(indices, comp_times, width=width)
    comm = plt.bar(indices, comm_times, width=width, bottom=comp_times)

    plt.legend((comp, comm), ('Comp', 'Comm'))
    plt.xticks(indices, ['W%d' % i for i in range(1, len(indices)+1)])
    plt.yticks(np.arange(0, 301, 50))
    plt.ylabel('Comp/Comm time (sec)')
    plt.show()


def main():
    #plot_test_accuracy_by_num_minibatches()
    #plot_comp_and_comm_time()


if __name__ == "__main__":
    main()
