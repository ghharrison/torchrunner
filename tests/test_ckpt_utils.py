import csv
import random
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pytest
import torch
import yaml
from torch.optim import SGD

import src.experiment_runner.ckpt_utils as ckpt
from src.experiment_runner.mlp import MLPModule
from src.experiment_runner.experiment_config import (
    ExperimentConfig, ModuleConfig
)


def test_log_metrics(tmp_path: Path):
    """
    - Use a `pytest` temporary directory `tmp_path`
    - use log_metrics to log random values to file
    - use log_metrics again to log different random
      values to the existing file
    - Read from the file and assert that it matches
    """
    epoch: int = random.randint(0, 99)
    train_loss: float = random.uniform(0, 1)
    test_loss: float = random.uniform(0, 1)
    expected = {
        'epoch': epoch,
        'train_loss': train_loss,
        'test_loss': test_loss,
    }
    metrics_file: Path = ckpt.log_metrics(
        run_dir=tmp_path,
        **expected
    )
    # Assert file created
    assert metrics_file.exists()

    # Assert CSV values match expected
    with open(metrics_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        assert set(reader.fieldnames) == expected.keys()
        row: Mapping[str, str] = reader.__next__()
        for key, value in expected:
            assert row[key] == value


def test_save_state(tmp_path):
    """
    - Create a temporary run dir and a temporary
      checkpoint dir inside it
    - Create an arbitrary set of values for state
      (model, optimizer, epoch, config, run_dir,
       checkpoint_dir)
    - Save a checkpoint file
    - Assert that saving to the same file raises a `FileExistsError`
    - Read the checkpoint and assert all the values match
    """
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    mlp_params = {
        'image_dim': 28,
        'linear_layer_size': 512,
        'output_classes': 10,
    }
    mlp_config = ModuleConfig(type='mlp', params=mlp_params)
    mlp = MLPModule(**mlp_params)

    sgd_params = {'lr': 0.001}
    sgd_config = ModuleConfig(type='sgd', params=sgd_params)
    sgd = SGD(mlp.parameters())

    loss_config = ModuleConfig(type='crossentropy', params={})
    loss_fn = torch.nn.CrossEntropyLoss()

    config = ExperimentConfig(
        model=mlp_config,
        optimizer=sgd_config,
        loss_fn=loss_config,
    )
    state = {
        'model': mlp,
        'optimizer': sgd,
        'loss_fn': loss_fn,
        'epoch': 0,
        'config': config,
        'run_dir': tmp_path,
        'checkpoint_dir': checkpoint_dir,
    }
    cfg_path, ckpt_path = ckpt.save_state(**state)
    with pytest.raises(FileExistsError):
        ckpt.save_state(**state)

    checkpoint = ckpt.load_checkpoint(ckpt_path)

    # State dict equality checks
    assert checkpoint.model_state_dict == mlp.state_dict()
    assert checkpoint.optimizer_state_dict == sgd.state_dict()
    assert checkpoint.loss_state_dict == loss_fn.state_dict()

    # Config equality checks
    assert checkpoint.config == asdict(config)
    with open(cfg_path, "r") as f:
        yaml_config = yaml.safe_load(f)
        assert yaml_config == checkpoint.config


def test_restore_state():
    """
    - Create state dicts for optimizer, model, RNG state
    - Create and save a temporary checkpoint
    - restore state and assert equality between state dicts and
      equality of each aspect of RNG state
    """
