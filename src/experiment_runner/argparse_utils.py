import argparse


def parse_args():
    """
    Parse arguments. `--config` is required and provides the base config;
    all other arguments are overrides.
    """
    parser = argparse.ArgumentParser()

    common_group = parser.add_argument_group('common', 'common arguments')
    common_group.add_argument('--config', type=str, required=True)

    return parser.parse_args()
