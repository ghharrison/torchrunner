import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import csv
import numpy as np
import torch
import yaml
from torch.optim import Optimizer
from torch.nn import Module

from experiment_config import ExperimentConfig


CONFIG_FILENAME: str = "config.yaml"


@dataclass(frozen=True)
class RNGState:
    python: Tuple = field(default_factory=tuple)
    numpy: Mapping = field(default_factory=dict)
    torch: torch.Tensor = field(default_factory=torch.Tensor)
    cuda: List[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True)
class RunCheckpoint:
    epoch: int
    config: Mapping = field(default_factory=dict)
    model_state_dict: Mapping = field(default_factory=dict)
    optimizer_state_dict: Mapping = field(default_factory=dict)
    rng_state: Mapping = field(default_factory=dict)


def save_state(
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        config: ExperimentConfig,
        run_dir: Path,
        checkpoint_dir: Path
) -> None:
    """
    Save the following to the run directory:
    - experiment config (YAML)
    - model checkpoints directory (epoch checkpoints and best)
    - metrics(?)
    - logs
    """

    # dump config to YAML
    config_dict: dict[str, Any] = asdict(config)
    config_path: Path = run_dir / CONFIG_FILENAME
    with open(config_path, 'w') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)

    # record RNG state
    rng = RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(legacy=False),
        torch=torch.random.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )

    # NumPy arrays are not trusted by `torch.load`; convert to Tensor
    rng.numpy['state']['key'] = torch.from_numpy(rng.numpy['state']['key'])

    # save model checkpoint
    checkpoint_path: Path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
    save_checkpoint(config, model, optimizer, epoch, rng, checkpoint_path)


def save_checkpoint(
        config: ExperimentConfig,
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        rng_state: RNGState,
        checkpoint_path: str,
) -> None:
    if os.path.exists(checkpoint_path):
        raise FileExistsError(f"Checkpoint {checkpoint_path} exists.")

    checkpoint = RunCheckpoint(
        config=asdict(config),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
        rng_state=asdict(rng_state),
    )
    torch.save(asdict(checkpoint), checkpoint_path)


def load_checkpoint(
        checkpoint_path: str,
) -> RunCheckpoint:
    checkpoint_dict: Mapping = torch.load(checkpoint_path)
    checkpoint: RunCheckpoint = RunCheckpoint(**checkpoint_dict)
    return checkpoint


def restore_rng_state(rng_dict: Mapping):
    rng_state = RNGState(**rng_dict)

    # convert serialized tensor back into ndarray
    rng_state.numpy['state']['key'] = rng_state.numpy['state']['key'].numpy()

    random.setstate(rng_state.python)
    np.random.set_state(rng_state.numpy)
    torch.random.set_rng_state(rng_state.torch)
    torch.cuda.set_rng_state_all(rng_state.cuda)


def restore_state_from_checkpoint(
        model: Module,
        optimizer: Optimizer,
        checkpoint: RunCheckpoint,
) -> None:
    model.load_state_dict(checkpoint.model_state_dict)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    restore_rng_state(checkpoint.rng_state)


def log_metrics(run_dir, epoch, train_loss, test_loss) -> None:
    metrics_file: Path = run_dir / "metrics.csv"
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "test_loss"],
            )
            writer.writeheader()

    with open(metrics_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "test_loss"],
        )
        writer.writerow({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
        })
