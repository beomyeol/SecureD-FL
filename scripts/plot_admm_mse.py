import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
    num_iterations = np.arange(1, 8)
    mses = [
        0.006531126192,
        1.62776E-09,
        6.59663E-12,
        3.70687E-14,
        3.36669E-16,
        3.91176E-16,
        1.09608E-16,
        1.09941E-16,
    ]

    stdevs = [
        0.000949984086,
        3.17239E-10,
        7.20424E-13,
        3.23782E-15,
        1.09715E-16,
        2.12642E-16,
        5.95880E-17,
        5.98702E-17,
    ]

    plt.plot(num_iterations, mses)

if __name__ == "__main__":
    main()
