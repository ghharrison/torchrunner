import pytest
import torch
from torch.optim import SGD

from experiment_runner.ckpt_utils import save_state
from experiment_runner.experiment_config import ModuleConfig, ExperimentConfig
from experiment_runner.mlp import MLPModule


@pytest.fixture
def mlp_setup():
    mlp_params = {
        'image_dim': 28,
        'linear_layer_size': 512,
        'output_classes': 10,
    }
    mlp_config = ModuleConfig(type='mlp', params=mlp_params)
    mlp = MLPModule(**mlp_params)
    return mlp_config, mlp


@pytest.fixture
def sgd_setup(mlp_setup):
    _, mlp = mlp_setup
    sgd_params = {'lr': 0.001}
    sgd_config = ModuleConfig(type='sgd', params=sgd_params)
    sgd = SGD(mlp.parameters())
    return sgd_config, sgd


@pytest.fixture
def loss_setup():
    loss_config = ModuleConfig(type='crossentropy', params={})
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_config, loss_fn


@pytest.fixture
def experiment_config(tmp_path, mlp_setup, sgd_setup, loss_setup):
    mlp_config, _ = mlp_setup
    sgd_config, _ = sgd_setup
    loss_config, _ = loss_setup
    config = ExperimentConfig(
        experiments_root=str(tmp_path),
        model=mlp_config,
        optimizer=sgd_config,
        loss_fn=loss_config,
    )
    return config


@pytest.fixture
def make_state(tmp_path, mlp_setup, sgd_setup, loss_setup, experiment_config):
    _, mlp = mlp_setup
    _, sgd = sgd_setup
    _, loss_fn = loss_setup

    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint"
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
def make_checkpoint(make_state):
    state = make_state
    cfg_path, ckpt_path = save_state(**state)
    return cfg_path, ckpt_path
