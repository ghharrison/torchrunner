"""Utility functions for managing experiment checkpoints.

Also defines dataclasses representing a training checkpoint and the current RNG
state.
"""

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
from experiment_runner.loss import CallableLoss, has_state_dict


CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class RNGState:
    """
    Dataclass representing state of the random number generator (RNG).
    Should be initialized via `RNGState.capture()` instead of constructed
    normally.

    Attributes:
        python: A tuple representing RNG state of `random` module.
        numpy: A dict representing RNG state of `np.random` module.
        torch: A Tensor representing RNG state of `torch` module.
        cuda: A list of Tensors representing RNG state of `torch.cuda` module.
    """
    python: tuple[Any, ...]
    numpy: dict[str, Any]
    torch: torch.Tensor
    cuda: list[torch.Tensor]

    @classmethod
    def capture(cls) -> Self:
        """
        Build an instance of `RNGState` whose attributes
        contain the current RNG state.

        Returns:
            An instance of `RNGState` with its `python`, `numpy`,
            `torch`, and `cuda` attributes set to the current RNG
            state of the `random`, `np.random`, `torch`, and `torch.cuda`
            modules in their corresponding formats.
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
        """
        Custom definition for equality between `RNGState` instances.
        Equality check is performed using `torch.allclose` as exact
        equality between `torch.Tensor` objects is unreliable due to
        precision errors.
        """
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
    """
    Dataclass representing a checkpoint of a training run.

    Attributes:
        epoch: An integer representing the last completed training epoch.
        model_state_dict: The `state_dict` of a `torch.nn.Module` model.
        optimizer_state_dict: The `state_dict` of a `torch.optim.Optimizer`.
        loss_state_dict: The `state_dict` of the loss function, if one exists.
           Defaults to an empty dictionary if no state dict is defined.
        rng_state: Dictionary representation of a `RNGState`.
    """
    epoch: int
    config: dict[str, Any]
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    loss_state_dict: dict[str, Any]
    rng_state: dict[str, Any]


def save_config(config: ExperimentConfig, run_dir: Path) -> Path:
    """
    Save the experiment config for the current training run to a YAML file.

    Args:
        config: an `ExperimentConfig` representing this experiment's
            configuration.

        run_dir: A `pathlib.Path` representing the directory for the current
            training run.

    Returns:
        A `pathlib.Path` pointing to the created config file.
    """
    config_dict = config.model_dump()
    config_path = run_dir / CONFIG_FILENAME
    if os.path.exists(config_path):
        raise FileExistsError(f"Config file {config_path} exists.")
    with open(config_path, 'w') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)

    return config_path


def save_state(
        model: Module,
        optimizer: Optimizer,
        loss_fn: CallableLoss,
        epoch: int,
        config: ExperimentConfig,
        checkpoint_dir: Path
) -> tuple[Path, Path]:
    """
    Save the current training state to a file.

    Args:
        model: An `nn.Module` instance representing the model being trained.
        optimizer: A `torch.optim.Optimizer` used to optimize the model.
        loss_fn: A loss function, expected to be callable and to return a
            `torch.Tensor`.
        epoch: An integer representing the current epoch.
        config: An `ExperimentConfig` containing the current experiment's
            configuration.
        checkpoint_dir: A `pathlib.Path` representing the directory storing
            checkpoints.

    Returns:
        A `pathlib.Path` pointing to the the saved checkpoint file.
    """

    # capture RNG state
    rng = RNGState.capture()
    # NumPy arrays are not trusted by `torch.load`; convert to Tensor
    rng.numpy['state']['key'] = torch.from_numpy(
        rng.numpy['state']['key']
    )

    # construct and save checkpoint to file
    if has_state_dict(loss_fn):
        loss_state_dict = loss_fn.state_dict()
    else:
        loss_state_dict = {}
    checkpoint = RunCheckpoint(
        config=config.model_dump(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        loss_state_dict=loss_state_dict,
        epoch=epoch,
        rng_state=asdict(rng),
    )
    checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
    checkpoint_path = save_checkpoint(
        run_checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    return checkpoint_path


def save_checkpoint(
        run_checkpoint: RunCheckpoint,
        checkpoint_path: Path,
) -> Path:
    """
    Save a `RunCheckpoint` object to file.

    Args:
        run_checkpoint: A `RunCheckpoint` instance.
        checkpoint_path: A destination `pathlib.Path`.

    Returns:
        A `pathlib.Path` to the saved checkpoint.

    Raises:
        `FileExistsError`: A checkpoint at the target path already exists.
    """
    if os.path.exists(checkpoint_path):
        raise FileExistsError(f"Checkpoint {checkpoint_path} exists.")

    torch.save(asdict(run_checkpoint), checkpoint_path)
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
    """
    Restore the training state from a `RunCheckpoint` object. This
    restores state dicts (where applicable) and the recorded RNG state.

    Args:
        model: The `nn.Module` being trained.
        optimizer: The model's optimizer.
        loss_fn: A callable loss function that may or may not have
            a `state_dict()` method.
        checkpoint: The `RunCheckpoint` to restore from.
    """
    model.load_state_dict(checkpoint.model_state_dict)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)
    if has_state_dict(loss_fn):
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
    """
    Record an epoch's train and test loss metrics to a CSV file.
    This will append a row to an existing CSV file or create a file
    if none exists. The CSV will always be named `metrics.csv` and will
    be underneath the run directory.

    Args:
        run_dir: A `pathlib.Path` containing the current run's artifacts.
        epoch: An integer representing the current epoch number.
        train_loss: A float scalar of the epoch's training loss.
        test_loss: A float scalar of the epoch's evaluation loss.

    Returns:
        A `pathlib.Path` to the created/modified CSV file.
    """
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
