import torch
from torch import nn

from experiment_runner.experiment_config import ModuleConfig


OPTIMIZERS = {
    "sgd": torch.optim.SGD,
}


def create_optimizer(
        model: nn.Module,
        optimizer_cfg: ModuleConfig,
):
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
