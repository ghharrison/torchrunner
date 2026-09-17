import pytest
from torch.optim import Optimizer

from experiment_runner.optimizer import create_optimizer
from experiment_runner.experiment_config import ModuleConfig


def test_get_device(experiment_config, mlp_setup):
    _, mlp = mlp_setup
    optimizer: Optimizer = create_optimizer(
        mlp,
        experiment_config.optimizer,
    )
    assert isinstance(optimizer, Optimizer)
    bogus_type_config = ModuleConfig(
        type="bogus", params={}
    )
    malformed_params_config = ModuleConfig(
        type="sgd", params={"bogus": None}
    )
    with pytest.raises(ValueError):
        create_optimizer(mlp, bogus_type_config)
    with pytest.raises(ValueError):
        create_optimizer(mlp, malformed_params_config)
