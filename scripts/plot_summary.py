import argparse
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


def load_events(path, name):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    events = ea.Scalars(name)

    return [(event.step, event.value) for event in events]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, nargs='+',
                        help='TF summary log dir(s)')
    parser.add_argument('--label', required=True, nargs='+',
                        help='Label(s) on the plot')
    parser.add_argument('--name', required=True, help='Scalar name to read')
    parser.add_argument('--output', help='Output name.')

    args = parser.parse_args()

    markers = ["x", "+", "."]
    colors = ["r", "b", "g"]

    for idx, (logdir, label) in enumerate(zip(*[args.logdir, args.label])):
        print('label={}, logdir={}'.format(label, logdir))

        events = load_events(logdir, args.name)

        steps, values = tuple(zip(*events))

        plt.plot(steps, values, label=label,
                 marker=markers[idx], color=colors[idx])

    plt.legend()

    if args.output:
        print('Saving the plot to {}...'.format(args.output))
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
