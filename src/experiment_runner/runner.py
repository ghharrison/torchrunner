import ckpt_utils
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from mlp import MLPModule
from optimizer import create_optimizer
from experiment_config import ExperimentConfig
from typing import List


class Runner:
    def __init__(
            self,
            experiment_config: ExperimentConfig,
    ):
        """
        Default constructor assumes new run
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # noqa

        self.config: ExperimentConfig = experiment_config
        self.device: torch.Device = device
        self.model: nn.Module = MLPModule().to(device)  # TODO: make configurable  # noqa
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()  # TODO: make configurable  # noqa
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

        self.experiments_root: Path = Path("experiments")
        self.experiment_dir: Path = self.experiments_root / self.config.experiment_name  # noqa
        self.run_dir: Path = make_run_dirs()
        self.checkpoint_dir: Path = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True)

        self.metrics: List = []

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path):
        """
        Alternative constructor assumes resuming from checkpoint
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # noqa
        checkpoint: ckpt_utils.RunCheckpoint = ckpt_utils.load_checkpoint(
            checkpoint_path=checkpoint_path,
        )
        config = ExperimentConfig.from_dict(checkpoint.config)
        runner = cls(device=device, config=config)
        runner.config = config
        runner.epoch = checkpoint.epoch + 1

        ckpt_utils.restore_state_from_checkpoint(
            runner.model, runner.optimizer, checkpoint,
        )

        return runner

    def train_step(self, train_loader: DataLoader):
        # Set the model to TRAIN mode
        self.model.train()
        size = len(train_loader.dataset)
        epoch_loss = 0

        for batch, (X, y) in enumerate(train_loader):
            # enumerating over pairs (X, y)
            # where X is batch input
            # and y is training output
            X = X.to(self.device)
            y = y.to(self.device)

            prediction = self.model(X)
            loss = self.loss_fn(prediction, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            batch_size = X.size(0)
            loss = loss.item()
            epoch_loss += loss * batch_size

            if batch % 100 == 0:
                current = batch * train_loader.batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        epoch_loss /= size
        print(f"Training Error: \n Avg loss: {epoch_loss:>8f} \n")
        return epoch_loss

    def eval_step(self, test_loader):
        # set the model to EVAL mode, which changes how some modules behave
        # Important ones are Batch normalization (since inference doesn't have
        # training minibatches) and Dropout (since randomly nixing connections
        # is undesirable during inference)

        self.model.eval()
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        test_loss, correct = 0, 0

        # We use the context manager `torch.no_grad()` to disable
        # gradient computation since we're no longer updating weights
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                prediction = self.model(X)

                # `.item()` converts a single-value tensor to a scalar
                test_loss += self.loss_fn(prediction, y).item()

                # "argmax" = "index of the maximum"
                batch_correct = prediction.argmax(1) == y
                correct += batch_correct.type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # noqa: E501

        return test_loss

    def train(self, train_loader, test_loader):
        epochs = 10
        for t in range(epochs):
            print(f"Epoch {t}\n-------------------------------")
            train_loss = self.train_step(train_loader)
            test_loss = self.eval_step(test_loader)
            ckpt_utils.save_state(
                config=self.config,
                model=self.model,
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
