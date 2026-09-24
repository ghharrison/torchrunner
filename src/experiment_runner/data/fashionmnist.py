import math

from experiment_runner.experiment_config import ExperimentConfig
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

BASE_TRANSFORMS = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]


def compute_stats(root="data", batch_size=64) -> tuple[float, float]:
    """
    Compute mean and standard deviation across the entire FashionMNIST
    dataset. This will be used to normalize the dataset.

    Args:
        root: A string of the directory the data is expected to be in, and
            where it will be downloaded to if not found.

        batch_size: Integer of batch size for the `DataLoader`. Should not
            affect the computed summary statistics.

        channels: Integer number of channels in the target dataset.

    Returns:
        A tuple of Tensors representing mean and standard deviation of each
        channel in the dataset.
    """
    data = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=v2.Compose(BASE_TRANSFORMS),
    )
    loader = DataLoader(data, batch_size=batch_size)

    mean = 0.0
    sq_mean = 0.0
    n_examples = 0

    for X, _ in loader:
        X: torch.Tensor 
        mean += X.mean().item() * batch_size
        sq_mean += (X**2).mean().item() * batch_size
        n_examples += batch_size
    mean /= n_examples
    sq_mean /= n_examples
    std = math.sqrt(sq_mean - math.pow(mean, 2))

    return mean, std


def compose_transforms(
    config: ExperimentConfig,
    train: bool,
) -> v2.Compose:
    if train:
        config_transforms = config.dataset.transforms.train
    else:
        config_transforms = config.dataset.transforms.eval

    transforms = []
    for transform in config_transforms:
        try:
            transform_cls = getattr(v2, transform.name)
        except AttributeError:
            # TODO: Allow definition and registration of custom transforms
            raise ValueError(
                f"Configured transform type {transform.name} is not present"
                f"in `torchvision.transforms.v2`."
            )
        if transform_cls is v2.Normalize and not transform.params:
            mean, std = compute_stats(
                config.dataset.root,
                config.batch_size,
            )
            transform.params = {"mean": mean, "std": std}
        transforms.append(transform_cls(**transform.params))
    return v2.Compose(transforms)


def create_loader(
        config: ExperimentConfig,
        root="data",
        train=True,
        download=True,
) -> DataLoader:
    transforms = compose_transforms(config, train)
    data = datasets.FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=transforms,
    )
    batch_size = config.batch_size
    loader = DataLoader(data, batch_size=batch_size, shuffle=train)
    return loader
