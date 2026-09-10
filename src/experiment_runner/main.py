import random
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from runner import Runner
from datasets.fashionmnist import create_loader
from experiment_config import ExperimentConfig
from argparse_utils import parse_args

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
OPTIMIZER_TYPE = "sgd"

OPTIMIZER_CONFIG = {
    "type": OPTIMIZER_TYPE,
    "params": {
        "lr": LEARNING_RATE,
    },
}

CONFIG = {
    "batch_size": 64,
    "optimizer": OPTIMIZER_CONFIG,
}


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


def main(run_source: NewRun | ResumeRun):
    match run_source:
        case NewRun(config):
            runner = Runner(
                experiment_config=config,
            )
        case ResumeRun(checkpoint_path):
            runner = Runner.from_checkpoint(checkpoint_path)

    train_dataloader = create_loader(
        train=True,
        batch_size=runner.config.batch_size
    )
    test_dataloader = create_loader(
        train=False,
        batch_size=runner.config.batch_size
    )
    runner.train(train_dataloader, test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        config_path = args.config
        config = ExperimentConfig.from_yaml(config_path)
        source = NewRun(config)
    else:
        if not args.checkpoint:
            raise RuntimeError("Must specify checkpoint or config")
        checkpoint_path = args.checkpoint
        source = ResumeRun(checkpoint_path)

    main(run_source=source)
