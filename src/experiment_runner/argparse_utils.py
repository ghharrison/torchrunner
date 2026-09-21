"""Argument parser for running experiments.

Expects exactly one of `--config` or `--checkpoint` to be specified
for starting a fresh experiment run or resuming from an existing checkpoint,
respectively. Resuming a previous training run will not modify the existing
run artifacts, but will continue in a new run.
"""

from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    """
    Parse arguments. `--config` is used to start a new experiment run
    with a given config. `--checkpoint` is used to resume an existing run.

    Returns
        An `argparse.Namespace` of the parsed arguments.
    """
    parser = ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--config', type=str)
    mode_group.add_argument('--checkpoint', type=str)

    return parser.parse_args()
