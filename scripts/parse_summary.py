import argparse

from tensorboard.backend.event_processing import event_accumulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True, help='TF summary log dir')
    parser.add_argument('--name', required=True, help='Scalar names to read')

    args = parser.parse_args()

    ea = event_accumulator.EventAccumulator(args.logdir)
    ea.Reload()

    events = ea.Scalars(args.name)

    for event in events:
        print('{} {}'.format(event.step, event.value))


if __name__ == "__main__":
    main()
