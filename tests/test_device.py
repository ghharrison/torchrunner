import pytest
import torch

from experiment_runner.device import get_device


def test_get_device(experiment_config):
    device: torch.device = get_device(experiment_config)
    assert device.type == "cpu"
    experiment_config.device = "bogus"
    with pytest.raises(ValueError):
        device = get_device(experiment_config)
