import argparse
import matplotlib.pyplot as plt
import numpy as np

from tensorboard.backend.event_processing import event_accumulator

_LABELS = {
    'train_loss': 'Loss',
    'train_accuracy': 'Accuracy',
    'validation_accuracy (after aggr.)': 'Accuracy',
}

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
    parser.add_argument('--yticks',
                        help='comma separate range (start,end,step)')

    args = parser.parse_args()

    markers = [".", "x", "+"]
    colors = ["g", "r", "b"]

    fontsize = 14

    for idx, (logdir, label) in enumerate(zip(*[args.logdir, args.label])):
        print('label={}, logdir={}'.format(label, logdir))

        name = args.name
        if 'local_only' in logdir and '(after aggr.)' in name:
            name = name[:name.find('(after aggr.)')].rstrip()

        events = load_events(logdir, name)

        steps, values = tuple(zip(*events))

        plt.plot(steps, values, label=label,
                 marker=markers[idx], color=colors[idx])

    plt.xlabel('Round', fontsize=fontsize)
    plt.ylabel(_LABELS[args.name], fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    if args.yticks:
        start, end, step = map(lambda x: float(x), args.yticks.split(','))
        plt.yticks(np.arange(start, end, step), fontsize=fontsize)
    else:
        plt.yticks(fontsize=fontsize)
    plt.tight_layout()

    if 'accuracy' in args.name:
        legend_loc = 'lower right'
    else:
        legend_loc = 'best'
    plt.legend(loc=legend_loc, fontsize=fontsize)

    if args.output:
        print('Saving the plot to {}...'.format(args.output))
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
