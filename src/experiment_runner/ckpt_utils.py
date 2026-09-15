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
from loss import CallableLoss


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
    loss_state_dict: Mapping = field(default_factory=dict)
    rng_state: Mapping = field(default_factory=dict)


def save_state(
        model: Module,
        optimizer: Optimizer,
        loss_fn: CallableLoss,
        epoch: int,
        config: ExperimentConfig,
        run_dir: Path,
        checkpoint_dir: Path
) -> Tuple[Path, Path]:
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
    if os.path.exists(config_path):
        raise FileExistsError(f"Config file {config_path} exists.")

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
    checkpoint_path = save_checkpoint(
        config=config,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch=epoch,
        rng_state=rng,
        checkpoint_path=checkpoint_path,
    )
    return config_path, checkpoint_path


def save_checkpoint(
        config: ExperimentConfig,
        model: Module,
        optimizer: Optimizer,
        loss_fn: CallableLoss,
        epoch: int,
        rng_state: RNGState,
        checkpoint_path: Path,
) -> Path:
    if os.path.exists(checkpoint_path):
        raise FileExistsError(f"Checkpoint {checkpoint_path} exists.")

    if hasattr(loss_fn, "state_dict"):
        if not callable(getattr(loss_fn, "state_dict")):
            raise ValueError(
                f"Loss function {loss_fn} had unexpected"
                f" non-callable attribute `state_dict`"
            )
        loss_state_dict = loss_fn.state_dict()
    else:
        loss_state_dict = {}
    checkpoint = RunCheckpoint(
        config=asdict(config),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        loss_state_dict=loss_state_dict,
        epoch=epoch,
        rng_state=asdict(rng_state),
    )
    torch.save(asdict(checkpoint), checkpoint_path)
    return checkpoint_path


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
        loss_fn: CallableLoss,
        checkpoint: RunCheckpoint,
) -> None:
    model.load_state_dict(checkpoint.model_state_dict)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    if hasattr(loss_fn, 'load_state_dict'):
        if not callable(getattr(loss_fn, 'load_state_dict')):
            raise ValueError(
                f"Loss function {loss_fn} had unexpected non-callable"
                f" attribute `load_state_dict`"
            )
        loss_fn.load_state_dict(checkpoint.loss_state_dict)
    restore_rng_state(checkpoint.rng_state)


def log_metrics(
        run_dir: Path,
        epoch: int,
        train_loss: float,
        test_loss: float,
) -> Path:
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

    return metrics_file
