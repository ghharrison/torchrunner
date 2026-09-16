import pytest
import torch
from torch.optim import SGD

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
    return mlp_params, mlp_config, mlp


@pytest.fixture
def sgd_setup(mlp_setup):
    _, _, mlp = mlp_setup
    sgd_params = {'lr': 0.001}
    sgd_config = ModuleConfig(type='sgd', params=sgd_params)
    sgd = SGD(mlp.parameters())
    return sgd_params, sgd_config, sgd


@pytest.fixture
def loss_setup():
    loss_config = ModuleConfig(type='crossentropy', params={})
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_config, loss_fn


@pytest.fixture
def experiment_config(mlp_setup, sgd_setup, loss_setup):
    _, mlp_config, _ = mlp_setup
    _, sgd_config, _ = sgd_setup
    _, loss_config, _ = loss_setup
    config = ExperimentConfig(
        model=mlp_config,
        optimizer=sgd_config,
        loss_fn=loss_config,
    )
    return config


@pytest.fixture
def make_state(tmp_path_factory, mlp_setup, sgd_setup, loss_setup):
    mlp_params, mlp_config, mlp = mlp_setup
    sgd_params, sgd_config, sgd = sgd_setup
    loss_config, loss_fn = loss_setup

    config = ExperimentConfig(
        model=mlp_config,
        optimizer=sgd_config,
        loss_fn=loss_config,
    )

    run_dir = tmp_path_factory.mktemp("run")
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir()
    state = {
        'model': mlp,
        'optimizer': sgd,
        'loss_fn': loss_fn,
        'epoch': 0,
        'config': config,
        'run_dir': run_dir,
        'checkpoint_dir': checkpoint_dir,
    }
    return config, state
