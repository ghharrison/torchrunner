"""
Main entrypoint for the experiment runner.

Invoked via `python -m experiment_runner` followed by command line args.
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_runner.argparse_utils import parse_args
from experiment_runner.data.common import create_loader
from experiment_runner.experiment_config import ExperimentConfig
from experiment_runner.runner import Runner


def set_all_seeds(seed: float) -> None:
    """
    Set all RNG seeds to the given value and attempt to run deterministically.

    Args:
        seed: A float to be used as the RNG seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    """
    Run, using either a config for a new run or a checkpoint path
    to continue from an existing checkpoint.

    Args:
        run_source: An `ExperimentConfig` or `Path`.
    """
    args = parse_args()
    if args.config:
        config_path: str = args.config
        source = ExperimentConfig.from_yaml(config_path)
    else:
        if not args.checkpoint:
            raise RuntimeError("Must specify checkpoint or config")
        source = Path(args.checkpoint)

    match source:
        case ExperimentConfig():
            runner = Runner(
                config=source,
            )
        case Path():
            runner = Runner.from_checkpoint(source)
        case _:
            raise TypeError(
                f"Input source was of unsupported type {type(source)}",
            )

    train_dataloader: DataLoader = create_loader(
        train=True,
        config=runner.config,
    )
    test_dataloader: DataLoader = create_loader(
        train=False,
        config=runner.config,
    )
    runner.train(train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
