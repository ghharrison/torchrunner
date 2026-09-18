import random
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiment_runner.argparse_utils import parse_args
from experiment_runner.data.fashionmnist import create_loader
from experiment_runner.experiment_config import ExperimentConfig
from experiment_runner.runner import Runner


@dataclass(frozen=True)
class NewRun:
    config: ExperimentConfig


@dataclass(frozen=True)
class ResumeRun:
    checkpoint_path: Path


def set_all_seeds(seed: float):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(run_source: NewRun | ResumeRun) -> None:
    match run_source:
        case NewRun(config):
            runner = Runner(
                config=config,
            )
        case ResumeRun(checkpoint_path):
            runner = Runner.from_checkpoint(checkpoint_path)

    train_dataloader: DataLoader = create_loader(
        train=True,
        batch_size=runner.config.batch_size
    )
    test_dataloader: DataLoader = create_loader(
        train=False,
        batch_size=runner.config.batch_size
    )
    runner.train(train_dataloader, test_dataloader)


if __name__ == "__main__":
    args: Namespace = parse_args()
    if args.config:
        config_path: str = args.config
        config = ExperimentConfig.from_yaml(config_path)
        source = NewRun(config)
    else:
        if not args.checkpoint:
            raise RuntimeError("Must specify checkpoint or config")
        checkpoint_path: str = args.checkpoint
        source = ResumeRun(checkpoint_path)

    main(run_source=source)
