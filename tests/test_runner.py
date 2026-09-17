import os
from dataclasses import asdict

import torch

from experiment_runner.ckpt_utils import RunCheckpoint
from experiment_runner.runner import Runner


def test_create_runner(experiment_config):
    """
    Construct a `Runner` object. Check:
    - Attributes are initialized properly
    - Run dir and checkpoint dir exist
    """
    runner = Runner(config=experiment_config)
    # runner.checkpoint_dir will be
    # cfg.experiments_root/cfg.experiment_name/[timestamp]/checkpoints

    assert runner.config == experiment_config
    assert runner.epoch == 0
    assert os.path.exists(runner.checkpoint_dir)


def test_create_from_checkpoint(make_checkpoint):
    """
    Construct a `Runner` using a predefined checkpoint
    file.
    - Check that epoch is checkpoint's recorded epoch + 1
    - Other functionality is covered by restore_state_from_checkpoint
    """
    cfg_path, checkpoint_path = make_checkpoint
    runner = Runner.from_checkpoint(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path)
    checkpoint = RunCheckpoint(**checkpoint_dict)

    assert checkpoint.epoch + 1 == runner.epoch
    assert checkpoint.config == asdict(runner.config)
