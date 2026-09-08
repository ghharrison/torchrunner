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


class Checkpointer:
    def __init__(self, config: ExperimentConfig, run_dir: Path):
        self.config: ExperimentConfig = config
        self.run_dir: Path = run_dir
        self.checkpoint_dir: Path = run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True)

    def save_state(self,
                   model: Module,
                   optimizer: Optimizer,
                   epoch: int):
        """
        Save the following to the run directory:
        - experiment config (YAML)
        - model checkpoints directory (epoch checkpoints and best)
        - metrics(?)
        - logs
        """

        # dump config to YAML
        config_dict = dict(self.config)
        config_path = self.run_dir / CONFIG_FILENAME
        with open(config_path, 'w') as config_file:
            yaml.dump(config_dict, config_file, default_flow_style=False)

        # save model checkpoint
        checkpoint_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pt'
        self.save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # TODO: export metrics
        pass

    def save_checkpoint(
            self,
            model: Module,
            optimizer: Optimizer,
            epoch: int,
            checkpoint_path: str,
    ):
        if os.path.exists(checkpoint_path):
            raise FileExistsError(f"Checkpoint {checkpoint_path} exists.")

        checkpoint = RunCheckpoint(
            config=dict(self.config),
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch,
        )
        torch.save(dict(checkpoint), checkpoint_path)
