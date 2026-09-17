from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.optim import SGD, Optimizer

from experiment_runner.ckpt_utils import save_state
from experiment_runner.experiment_config import ModuleConfig, ExperimentConfig
from experiment_runner.mlp import MLPModule


@pytest.fixture
def mlp_config() -> ModuleConfig:
    mlp_params = {
        'image_dim': 28,
        'linear_layer_size': 512,
        'output_classes': 10,
    }
    return ModuleConfig(type='mlp', params=mlp_params)


@pytest.fixture
def mlp(mlp_config: ModuleConfig) -> nn.Module:
    return MLPModule(**mlp_config.params)


@pytest.fixture
def sgd_config() -> ModuleConfig:
    sgd_params = {'lr': 0.001}
    return ModuleConfig(type='sgd', params=sgd_params)


@pytest.fixture
def sgd(mlp: MLPModule, sgd_config: ModuleConfig) -> Optimizer:
    sgd = SGD(mlp.parameters(), **sgd_config.params)
    return sgd


@pytest.fixture
def loss_config() -> ModuleConfig:
    return ModuleConfig(type='crossentropy', params={})


@pytest.fixture
def loss_fn(loss_config: ModuleConfig) -> nn.Module:
    return torch.nn.CrossEntropyLoss(**loss_config.params)


@pytest.fixture
def experiment_config(
    tmp_path: Path,
    mlp_config: ModuleConfig,
    sgd_config: ModuleConfig,
    loss_config: ModuleConfig,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiments_root=str(tmp_path),
        model=mlp_config,
        optimizer=sgd_config,
        loss_fn=loss_config,
    )


@pytest.fixture
def make_state(
    tmp_path: Path,
    mlp: nn.Module,
    sgd: Optimizer,
    loss_fn: nn.Module,
    experiment_config: ExperimentConfig,
) -> dict[str, Any]:

    run_dir: Path = tmp_path / "run"
    checkpoint_dir: Path = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    state = {
        'model': mlp,
        'optimizer': sgd,
        'loss_fn': loss_fn,
        'epoch': 0,
        'config': experiment_config,
        'run_dir': run_dir,
        'checkpoint_dir': checkpoint_dir,
    }
    return state


@pytest.fixture
def make_checkpoint(make_state) -> tuple[Path, Path]:
    cfg_path, ckpt_path = save_state(**make_state)
    return cfg_path, ckpt_path
