import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pathlib import Path


class Runner:
    def __init__(
            self,
            device: torch.Device,
            model: nn.Module,
            loss_fn: nn.Module,
            optimizer: Optimizer,
    ):
        self.device: torch.Device = device
        self.model: nn.Module = model.to(device)
        self.loss_fn: nn.Module = loss_fn
        self.optimizer: Optimizer = optimizer

        self.experiment_dir: Path = Path("experiments") / "fashion_mnist_001"
        self.checkpoint_dir: Path = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, epoch: int):
        torch.save(
            self.model.state_dict(),
            self.checkpoint_dir / f'mlpmodule_weights_{epoch}.pth'
        )

    def train_loop(self, train_loader: DataLoader):
        # Set the model to TRAIN mode
        self.model.train()
        size = len(train_loader.dataset)

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
        if batch % 100 == 0:
            loss = loss.item()
            current = batch * train_loader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def eval_loop(self, test_loader):
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
