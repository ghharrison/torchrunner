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
            f"Unsupported optimizer type {ModuleConfig.type}. "
            f"Current supported optimizers: {",".join(OPTIMIZERS.keys())}"
        )

    optimizer = optimizer_cls(model.parameters(), **optimizer_cfg.params)
    return optimizer
