import pytest

from experiment_runner.loss import get_loss_fn


def test_get_loss_fn(experiment_config):
    loss_fn = get_loss_fn(experiment_config)
    assert callable(loss_fn)

    experiment_config.loss_fn.params = {
        'bogus_param': None,
    }
    with pytest.raises(ValueError):
        get_loss_fn(experiment_config)
    experiment_config.loss_fn.type = "bogus_type"
    with pytest.raises(ValueError):
        get_loss_fn(experiment_config)
