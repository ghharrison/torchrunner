from typing import Mapping

from torch import nn

from experiment_config import ExperimentConfig

LOSS_FUNCTIONS: Mapping[str, nn.Module] = {
    "crossentropy": nn.CrossEntropyLoss,
}


def get_loss_fn(
        experiment_config: ExperimentConfig) -> nn.Module:
    loss_config = experiment_config.loss_fn
    loss_type = loss_config.type
    loss_params = loss_config.params

    try:
        module_cls: type = LOSS_FUNCTIONS[loss_type.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported loss function {loss_params}. "
            f"Current supported models: {",".join(LOSS_FUNCTIONS.keys())}"
        )

    try:
        loss_fn: nn.Module = module_cls(**loss_params)
    except TypeError as err:
        raise ValueError(
            f"Module {module_cls.__name__} could not be initialized "
            f"with parameters {loss_params}: {err}"
        )
    return loss_fn
