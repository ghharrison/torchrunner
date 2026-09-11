from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    """
    Parse arguments. `--config` is used to start a new experiment run
    with a given config. `--checkpoint` is used to resume an existing run.
    """
    parser: ArgumentParser = ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--config', type=str)
    mode_group.add_argument('--checkpoint', type=str)

    return parser.parse_args()
