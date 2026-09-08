import csv
import os
import torch
import yaml
from dataclasses import dataclass, field
from experiment_config import ExperimentConfig
from pathlib import Path
from typing import Mapping
from torch.optim import Optimizer
from torch.nn import Module

CONFIG_FILENAME = "config.yaml"


@dataclass
class RunCheckpoint:
    epoch: int
    config: Mapping = field(default_factory=dict)
    model_state_dict: Mapping = field(default_factory=dict)
    optimizer_state_dict: Mapping = field(default_factory=dict)

    def keys(self):
        return ['epoch', 'config', 'model_state_dict', 'optimizer_state_dict']

    def __getitem__(self, key):
        return getattr(self, key)


def save_state(
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        config: ExperimentConfig,
        run_dir: Path,
        checkpoint_dir: Path
):
    """
    Save the following to the run directory:
    - experiment config (YAML)
    - model checkpoints directory (epoch checkpoints and best)
    - metrics(?)
    - logs
    """

    # dump config to YAML
    config_dict = dict(config)
    config_path = run_dir / CONFIG_FILENAME
    with open(config_path, 'w') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)

    # save model checkpoint
    checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
    save_checkpoint(config, model, optimizer, epoch, checkpoint_path)


def save_checkpoint(
        config: ExperimentConfig,
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        checkpoint_path: str,
):
    if os.path.exists(checkpoint_path):
        raise FileExistsError(f"Checkpoint {checkpoint_path} exists.")

    checkpoint = RunCheckpoint(
        config=dict(config),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
    )
    torch.save(dict(checkpoint), checkpoint_path)


def load_checkpoint(
        checkpoint_path: str,
) -> RunCheckpoint:
    checkpoint_dict: Mapping = torch.load(checkpoint_path)
    checkpoint = RunCheckpoint(**checkpoint_dict)
    return checkpoint


def restore_state_from_checkpoint(
        model: Module,
        optimizer: Optimizer,
        checkpoint: RunCheckpoint,
):
    model.load_state_dict(checkpoint.model_state_dict)
    optimizer.load_state_dict(checkpoint.optimizer_state_dict)


def log_metrics(run_dir, epoch, train_loss, test_loss):
    metrics_file = run_dir / "metrics.csv"
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
