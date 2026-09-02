import torch
from torch import nn
from torch.optim import SGD
from runner.runner import Runner
from data.fashionmnist import get_loader
from models.mlp import MLPModule

LEARNING_RATE = 1e-3
BATCH_SIZE = 64


def main():
    train_dataloader = get_loader(train=True)
    test_dataloader = get_loader(train=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # noqa
    model = MLPModule()
    runner = Runner(
        device=device,
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=SGD(
            params=model.parameters(), lr=LEARNING_RATE,
        )
    )
    runner.train(train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
