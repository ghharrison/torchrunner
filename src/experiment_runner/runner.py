from datetime import datetime
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.types import Number

import experiment_runner.ckpt_utils as ckpt_utils
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
        Default constructor assumes new run
        """
        self.config: ExperimentConfig = config
        self.device: torch.Device = get_device(config)
        self.model: nn.Module = get_module(config).to(self.device)
        self.loss_fn: nn.Module = get_loss_fn(config).to(self.device)
        self.optimizer: Optimizer = create_optimizer(
            self.model,
            self.config.optimizer,
        )
        self.epoch: int = 0

        def make_run_dirs() -> Path:
            timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir: Path = self.experiment_dir / timestamp
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True)
            return run_dir

        self.experiments_root: Path = Path(config.experiments_root)
        self.experiment_dir: Path = (
            self.experiments_root / self.config.experiment_name
        )
        self.run_dir: Path = make_run_dirs()
        self.checkpoint_dir: Path = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> Runner:
        """
        Alternative constructor assumes resuming from checkpoint
        """
        print(f"Restoring training from checkpoint at {checkpoint_path}")
        checkpoint: ckpt_utils.RunCheckpoint = ckpt_utils.load_checkpoint(
            checkpoint_path=checkpoint_path,
        )
        config = ExperimentConfig.from_dict(
            checkpoint.config,
        )
        runner: Runner = cls(experiment_config=config)
        runner.epoch = checkpoint.epoch + 1

        print(f"resuming training from epoch {runner.epoch}")

        ckpt_utils.restore_state_from_checkpoint(
            runner.model, runner.optimizer, checkpoint,
        )

        return runner

    def train_step(self, train_loader: DataLoader) -> float:
        # Set the model to TRAIN mode
        self.model.train()
        size: int = len(train_loader.dataset)
        epoch_loss: Number = 0

        for batch, (X, y) in enumerate(train_loader):
            X: Tensor
            y: Tensor

            X = X.to(self.device)
            y = y.to(self.device)

            prediction: Tensor = self.model(X)
            loss: Tensor = self.loss_fn(prediction, y)

            # Backpropagation
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

    def eval_step(self, test_loader) -> float:
        # set the model to EVAL mode, which changes how some modules behave
        # Important ones are Batch normalization (since inference doesn't have
        # training minibatches) and Dropout (since randomly nixing connections
        # is undesirable during inference)

        self.model.eval()
        size: int = len(test_loader.dataset)
        num_batches: int = len(test_loader)
        test_loss: Number = 0
        correct: int = 0

        # We use the context manager `torch.no_grad()` to disable
        # gradient computation since we're no longer updating weights
        with torch.no_grad():
            for X, y in test_loader:
                X: Tensor = X.to(self.device)
                y: Tensor = y.to(self.device)
                prediction: Tensor = self.model(X)

                # `.item()` converts a single-value tensor to a scalar
                test_loss += self.loss_fn(prediction, y).item()

                # "argmax" = "index of the maximum"
                batch_correct: Tensor = (prediction.argmax(1) == y)
                correct += batch_correct.type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # noqa: E501

        return test_loss

    def train(self, train_loader, test_loader) -> None:
        start_epoch: int = self.epoch
        for t in range(start_epoch, self.config.epochs):
            print(f"Epoch {t}\n-------------------------------")
            train_loss: float = self.train_step(train_loader)
            test_loss: float = self.eval_step(test_loader)
            ckpt_utils.save_state(
                config=self.config,
                model=self.model,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                epoch=t,
                checkpoint_dir=self.checkpoint_dir,
                run_dir=self.run_dir,
            )
            ckpt_utils.log_metrics(
                run_dir=self.run_dir,
                epoch=t,
                train_loss=train_loss,
                test_loss=test_loss,
            )
            self.epoch = t
