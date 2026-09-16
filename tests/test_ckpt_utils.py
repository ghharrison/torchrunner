import copy
import csv
import random
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import pytest
import yaml
from torch.testing import assert_close

import experiment_runner.ckpt_utils as ckpt
from experiment_runner.ckpt_utils import RNGState


def test_log_metrics(tmp_path: Path) -> None:
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
        for key, value in expected.items():
            assert row[key] == str(value)


@pytest.fixture
def make_checkpoint(make_state):
    config, state = make_state
    cfg_path, ckpt_path = ckpt.save_state(**state)
    return cfg_path, ckpt_path


def test_save_state(
        mlp_setup,
        sgd_setup,
        loss_setup,
        make_state,
        make_checkpoint,
) -> None:
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
    mlp_params, mlp_config, mlp = mlp_setup
    sgd_params, sgd_config, sgd = sgd_setup
    loss_config, loss_fn = loss_setup

    config, state = make_state

    cfg_path, ckpt_path = make_checkpoint
    with pytest.raises(FileExistsError):
        ckpt.save_state(**state)

    checkpoint = ckpt.load_checkpoint(ckpt_path)

    # State dict equality checks
    for key, value in mlp.state_dict().items():
        assert_close(
            checkpoint.model_state_dict[key],
            value,
        )
    for key, value in sgd.state_dict().items():
        assert_close(
            checkpoint.optimizer_state_dict[key],
            value,
        )
    for key, value in loss_fn.state_dict().items():
        assert_close(
            checkpoint.loss_state_dict[key],
            value,
        )

    # Config equality checks
    assert checkpoint.config == asdict(config)
    with open(cfg_path, "r") as f:
        yaml_config = yaml.safe_load(f)
        assert yaml_config == checkpoint.config


def assert_state_close(expected: Mapping, actual: Mapping) -> None:
    for key, value in expected.items():
        assert_close(value, actual[key])


def test_restore_state(
        mlp_setup,
        sgd_setup,
        loss_setup,
        make_state,
        make_checkpoint,
):
    """
    - Create state dicts for optimizer, model, RNG state
    - Create and save a temporary checkpoint
    - restore state and assert equality between state dicts and
      equality of each aspect of RNG state
    """
    mlp_params, mlp_config, mlp = mlp_setup
    sgd_params, sgd_config, sgd = sgd_setup
    loss_config, loss_fn = loss_setup

    config, state = make_state

    expected_mlp_state = copy.deepcopy(mlp.state_dict())
    expected_sgd_state = copy.deepcopy(sgd.state_dict())
    expected_loss_state = copy.deepcopy(loss_fn.state_dict())

    expected_rng_state = copy.deepcopy(RNGState.capture())

    cfg_path, ckpt_path = make_checkpoint
    checkpoint = ckpt.load_checkpoint(ckpt_path)
    ckpt.restore_state_from_checkpoint(mlp, sgd, loss_fn, checkpoint)

    actual_mlp_state = mlp.state_dict()
    actual_sgd_state = sgd.state_dict()
    actual_loss_state = loss_fn.state_dict()

    assert_state_close(expected_mlp_state, actual_mlp_state)
    assert_state_close(expected_sgd_state, actual_sgd_state)
    assert_state_close(expected_loss_state, actual_loss_state)

    actual_rng_state = RNGState.capture()
    assert expected_rng_state == actual_rng_state
