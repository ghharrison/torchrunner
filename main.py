import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from mlp_module import MLPModule
from runner import Runner

LEARNING_RATE = 1e-3
BATCH_SIZE = 64


def main():
    training_data = datasets.FashionMNIST(
        root="data",  # root directory of dataset
        train=True,  # training or evaluation split(?)
        download=True,  # whether to download if not found
        transform=v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),  # a PIL image to transformed version function
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),
    )
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64)

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

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        runner.train_loop(train_dataloader)
        runner.eval_loop(test_dataloader)
        runner.save_checkpoint(t)


if __name__ == "__main__":
    main()
