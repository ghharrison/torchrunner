import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

from experiment_runner.experiment_config import ExperimentConfig


def compose_transforms(
    config: ExperimentConfig,
    train: bool,
) -> v2.Compose:
    """
    Compose a configurable sequence of transforms for this dataset.

    Args:
        config: The experiment's `ExperimentConfig`. The applied transforms
            will be under dataset.transforms.train or dataset.transforms.eval.

        train: A boolean for whether the transforms are for training (as
            opposed to evaluation.)

    Returns:
        A `v2.Compose` containing the configured transforms in the order they
        were listed in the config.
    """
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
        if 'dtype' in transform.params:
            target_type = transform.params['dtype']
            if hasattr(torch, target_type):
                target_type = getattr(torch, target_type)
                if type(target_type) is torch.dtype:
                    transform.params['dtype'] = target_type
                else:
                    raise ValueError(
                        f"Configured transform {transform.name} had argument "
                        f"`dtype` but {target_type} is not a `torch.dtype`"
                    )
            else:
                raise ValueError(
                    f"Configured transform {transform.name} is not recognized "
                    f"as a `torchvision.transforms.v2` transform"
                )
        transforms.append(transform_cls(**transform.params))
    return v2.Compose(transforms)


def create_loader(
        train: bool,
        config: ExperimentConfig,
) -> DataLoader:
    """
    Currently assumes that any configured transform is an attribute of
    `torchvision.transforms.v2` and that the configured dataset is a
    `torchvision.datasets.VisionDataset`.
    """
    config_transforms = getattr(
        config.dataset.transforms, 'train' if train else 'eval'
    )

    transforms = []
    for name, params in config_transforms.items():
        try:
            transform_cls = getattr(v2, name)
        except AttributeError:
            # TODO: Allow definition and registration of custom transforms
            raise ValueError(
                f"Configured transform type {name} is not present in "
                f"`torchvision.transforms.v2`."
            )
        transforms.append(transform_cls(**params))
    dataset_name = config.dataset.name
    if hasattr(datasets, ):
        dataset_cls: type = getattr(datasets, dataset_name)
        if not issubclass(dataset_cls, datasets.VisionDataset):
            raise ValueError(
                f"Configured dataset `{dataset_name}` exists in "
                "`torchvision.datasets` but is not a `VisionDataset`"
            )
    else:
        raise AttributeError(
            f"Configured dataset {dataset_name}`torchvision.datasets` was not "
            "found in `torchvision.datasets`"
        )
    data: Dataset = dataset_cls(
        root=config.dataset.root,
        train=train,
        download=True,
        transform=v2.Compose(transforms),
    )
    return DataLoader(data, batch_size=config.batch_size, shuffle=train)
