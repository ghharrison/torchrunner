"""
Creation of `torch.optim.Optimizer` for model training.
"""

import torch
from torch import nn
from torch.optim import Optimizer

from experiment_runner.experiment_config import ModuleConfig


OPTIMIZERS: dict[str, type[Optimizer]] = {
    "sgd": torch.optim.SGD,
}


def create_optimizer(
        model: nn.Module,
        optimizer_cfg: ModuleConfig,
) -> Optimizer:
    """
    Construct an optimizer for a model's parameters.

    Args:
        model: The `nn.Module` whose parameters will be optimized by the
            created optimizer.

        optimizer_cfg: a `ModuleConfig` defining the type and construction
            parameters of the desired optimizer.

    Returns: a `torch.optim.Optimizer`.

    Raises:
        `ValueError`: The configured optimizer is not a supported type, or
            the optimizer could not be initialized with the configured
            parameters.
    """
    try:
        optimizer_cls = OPTIMIZERS[optimizer_cfg.type.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported optimizer type {optimizer_cfg.type}. "
            f"Current supported optimizers: {",".join(OPTIMIZERS.keys())}"
        )
    try:
        optimizer = optimizer_cls(model.parameters(), **optimizer_cfg.params)
    except TypeError as err:
        raise ValueError(
            f"Could not create optimizer of type `{optimizer_cls.__name__}` "
            f"with parameters {optimizer_cfg.params}: {err}"
        )
    return optimizer
