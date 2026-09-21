"""
Utilities and types related to loss functions.

Defines a `CallableLoss` type that fits both simple loss
functions and `nn.Module` losses like `nn.CrossEntropyLoss`.
Also includes a `Protocol` and a `TypeGuard` to assist in static
type checking.
"""
from typing import Any, Callable, Mapping, TypeAlias, TypeGuard, Protocol

import torch
from torch import nn

from experiment_runner.experiment_config import ExperimentConfig

CallableLoss: TypeAlias = Callable[..., torch.Tensor]

LOSS_FUNCTIONS: Mapping[str, CallableLoss] = {
    "crossentropy": nn.CrossEntropyLoss,
}


class CallableLossWithStateDict(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        ...


def has_state_dict(
    loss_fn: CallableLoss,
) -> TypeGuard[CallableLossWithStateDict]:
    """
    Check that `loss_fn` has a callable `state_dict` and
    supports calls to `load_state_dict`.
    """
    return (
        hasattr(loss_fn, "state_dict")
        and callable(loss_fn.state_dict)
        and hasattr(loss_fn, "load_state_dict")
        and callable(loss_fn.load_state_dict)
    )


def get_loss_fn(
        experiment_config: ExperimentConfig) -> CallableLoss:
    """
    Construct a `CallableLoss` matching the loss type in an experiment
    config.

    Args:
        experiment_config: An `ExperimentConfig` for the current experiment.

    Returns:
        A `CallableLoss`, which is a `Callable[..., Tensor]`.
    """
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
        loss_fn: CallableLoss = module_cls(**loss_params)
    except TypeError as err:
        raise ValueError(
            f"Module {module_cls.__name__} could not be initialized "
            f"with parameters {loss_params}: {err}"
        )
    return loss_fn
