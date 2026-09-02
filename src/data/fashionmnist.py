import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader


def get_loader(root="data", train=True, download=True, batch_size=64):
    training_data = datasets.FashionMNIST(
        root=root, 
        train=train,
        download=download,
        transform=v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        ),
    )
    loader = DataLoader(training_data, batch_size=batch_size, shuffle=train)
    return loader
