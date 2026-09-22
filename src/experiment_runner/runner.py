"""
Runner class for handling experiment state.

Requires an `ExperimentConfig` or a path to a checkpoint
to initialize. Once created, calling `train()` with training and
evaluation `DataLoader` objects provided will handle the rest.
"""
from datetime import datetime
from pathlib import Path
from typing import Self

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.types import Number

import experiment_runner.ckpt_utils as ckpt_utils
from experiment_runner.ckpt_utils import RunCheckpoint
from experiment_runner.device import get_device
from experiment_runner.experiment_config import ExperimentConfig
from experiment_runner.model import get_module
from experiment_runner.loss import get_loss_fn
from experiment_runner.optimizer import create_optimizer


class Runner:
    def __init__(
            self,
            config: ExperimentConfig,
    ) -> None:
        """
        Default constructor. Used when starting a training run
        from scratch.

        Args:
            config: an `ExperimentConfig`.
        """
        self.config = config
        self.device = get_device(config)
        self.model = get_module(config).to(self.device)
        self.loss_fn = get_loss_fn(config).to(self.device)
        self.optimizer = create_optimizer(
            self.model,
            self.config.optimizer,
        )
        self.epoch = 0

        self.experiments_root: Path = Path(config.experiments_root)
        self.experiment_dir: Path = (
            self.experiments_root / self.config.experiment_name
        )
        self.run_dir: Path = self._create_run_dirs()
        ckpt_utils.save_config(self.config, self.run_dir)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> Self:
        """
        Alternative constructor. Initializes a new run as a continuation
        of an existing run (which is represented by a checkpoint file).

        Args:
            checkpoint_path: a `pathlib.Path` to the experiment checkpoint.
                (The checkpoint and other artifacts from the previous run
                will not be modified.)
        """
        print(f"Restoring training from checkpoint at {checkpoint_path}")
        checkpoint = ckpt_utils.load_checkpoint(
            checkpoint_path=checkpoint_path,
        )
        config = ExperimentConfig.from_dict(
            checkpoint.config,
        )
        runner = cls(config)
        runner._restore_from_checkpoint(checkpoint)
        return runner

    def _create_run_dirs(self) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self.experiment_dir / timestamp
        run_dir.mkdir(parents=True)
        self.checkpoint_dir = run_dir / "checkpoints"
        self.checkpoint_dir.mkdir()

        return run_dir

    def _restore_from_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        self.epoch = checkpoint.epoch + 1
        ckpt_utils.restore_state_from_checkpoint(
            self.model,
            self.optimizer,
            self.loss_fn,
            checkpoint,
        )
        print(f"resuming training from epoch {self.epoch}")

    def train_step(self, train_loader: DataLoader) -> float:
        """
        Do a single training step, updating model parameters and
        producing an epoch training loss.

        Args:
            train_loader: The training `DataLoader`.

        Returns:
            A `float` representing this epoch's training loss.
        """
        self.model.train()
        size = len(train_loader.dataset)
        epoch_loss = 0

        for batch, (X, y) in enumerate(train_loader):
            X: Tensor
            y: Tensor

            X = X.to(self.device)
            y = y.to(self.device)

            prediction: Tensor = self.model(X)
            loss: Tensor = self.loss_fn(prediction, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_size: int = X.size(0)
            loss_value: Number = loss.item()
            epoch_loss += loss_value * batch_size

            if batch % 100 == 0:
                current = batch * train_loader.batch_size + len(X)
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        epoch_loss /= size
        print(f"Training Error: \n Avg loss: {epoch_loss:>8f} \n")
        return epoch_loss

    def eval_step(self, test_loader: DataLoader) -> float:
        """
        Do a single evaluation step, updating model parameters and
        producing an epoch evaluation loss.

        Args:
            test_loader: The evaluation `DataLoader`.

        Returns:
            A `float` representing this epoch's evaluation loss.
        """
        self.model.eval()
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in test_loader:
                X: Tensor = X.to(self.device)
                y: Tensor = y.to(self.device)
                prediction: Tensor = self.model(X)

                test_loss += self.loss_fn(prediction, y).item()

                batch_correct: Tensor = (prediction.argmax(1) == y)
                correct += batch_correct.type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # noqa: E501

        return test_loss

    def train(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        """
        The main train/eval loop. Performs training and evaluation
        once per epoch for the number of epochs determined by the
        experiment config.

        Args:
            train_loader: the `DataLoader` for training data.
            test_loader: the 'DataLoader' for the evaluation data.
        """
        start_epoch = self.epoch
        for t in range(start_epoch, self.config.epochs):
            print(f"Epoch {t}\n-------------------------------")
            train_loss = self.train_step(train_loader)
            test_loss = self.eval_step(test_loader)
            ckpt_utils.save_state(
                config=self.config,
                model=self.model,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                epoch=t,
                checkpoint_dir=self.checkpoint_dir,
            )
            ckpt_utils.log_metrics(
                run_dir=self.run_dir,
                epoch=t,
                train_loss=train_loss,
                test_loss=test_loss,
            )
            self.epoch = t
