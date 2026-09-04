import os
import torch
import yaml
from experiment_config import ExperimentConfig
from pathlib import Path

CONFIG_FILENAME = "config.yaml"


class Checkpointer:
    def __init__(self, config: ExperimentConfig, run_dir: Path):
        self.config: ExperimentConfig = config
        self.run_dir: Path = run_dir

    def save_state(self, model: torch.nn.Module, epoch: int):
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
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / f'epoch_{epoch:03d}.pt'
        self.save_checkpoint(model, checkpoint_path)

        # TODO: save optimizer checkpoint
        pass

    def save_checkpoint(
            self,
            model: torch.nn.Module,
            checkpoint_path: str,
    ):
        if os.path.exists(checkpoint_path):
            raise FileExistsError(f"Checkpoint {checkpoint_path} exists.")
        torch.save(
            model.state_dict(),
            checkpoint_path,
        )
