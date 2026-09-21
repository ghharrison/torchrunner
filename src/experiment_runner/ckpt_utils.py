import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Self

import csv
import numpy as np
import torch
import yaml
from torch.optim import Optimizer
from torch.nn import Module

from experiment_runner.experiment_config import ExperimentConfig
from experiment_runner.loss import CallableLoss, has_load_state_dict


CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class RNGState:
    python: tuple[Any, ...]
    numpy: dict[str, Any]
    torch: torch.Tensor
    cuda: list[torch.Tensor]

    @classmethod
    def capture(cls) -> Self:
        """
        Build an instance of `RNGState` whose attributes
        contain the current RNG state.
        """
        return cls(
            python=random.getstate(),
            numpy=np.random.get_state(legacy=False),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
        )

    def restore(self) -> None:
        """
        Set the RNG state to match the values of this
        `RNGState` instance's attributes.
        """
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch)
        torch.cuda.set_rng_state_all(self.cuda)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RNGState):
            return NotImplemented

        return all([
            self.python == other.python,
            torch.allclose(
                torch.from_numpy(self.numpy['state']['key']),
                torch.from_numpy(other.numpy['state']['key']),
            ),
            torch.allclose(self.torch, other.torch),
            torch.allclose(torch.cat(self.cuda), torch.cat(other.cuda))
        ])


@dataclass(frozen=True)
class RunCheckpoint:
    epoch: int
    config: dict[str, Any]
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    loss_state_dict: dict[str, Any]
    rng_state: dict[str, Any]


def save_state(
        model: Module,
        optimizer: Optimizer,
        loss_fn: CallableLoss,
        epoch: int,
        config: ExperimentConfig,
        run_dir: Path,
        checkpoint_dir: Path
) -> tuple[Path, Path]:
    """
    Save the following to the run directory:
    - experiment config (YAML)
    - model checkpoints directory (epoch checkpoints and best)
    - metrics(?)
    - logs
    """

    # dump config to YAML
    config_dict = asdict(config)
    config_path = run_dir / CONFIG_FILENAME
    if os.path.exists(config_path):
        raise FileExistsError(f"Config file {config_path} exists.")

    with open(config_path, 'w') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)

    # record RNG state
    rng = RNGState.capture()
    # NumPy arrays are not trusted by `torch.load`; convert to Tensor
    rng.numpy['state']['key'] = torch.from_numpy(
        rng.numpy['state']['key']
    )

    # save model checkpoint
    checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
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
        checkpoint_path: Path,
) -> RunCheckpoint:
    return RunCheckpoint(**torch.load(checkpoint_path))


def restore_state_from_checkpoint(
        model: Module,
        optimizer: Optimizer,
        loss_fn: CallableLoss,
        checkpoint: RunCheckpoint,
) -> None:
    model.load_state_dict(checkpoint.model_state_dict)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    if has_load_state_dict(loss_fn):
        loss_fn.load_state_dict(checkpoint.loss_state_dict)

    rng_state = RNGState(**checkpoint.rng_state)

    # convert serialized tensor back into ndarray
    rng_state.numpy['state']['key'] = rng_state.numpy['state']['key'].numpy()

    rng_state.restore()


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
