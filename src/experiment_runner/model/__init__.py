from typing import Mapping

from torch import nn

import experiment_runner.model.mlp as mlp
import experiment_runner.model.cnn as cnn
from experiment_runner.experiment_config import ExperimentConfig

MODELS: Mapping[str, type[nn.Module]] = {
    "mlp": mlp.MLPBlock,
    "cnn": cnn.CNNBlock,
}


def get_module(
        experiment_config: ExperimentConfig) -> nn.Module:
    """
    Given an `ExperimentConfig` whose `model` field contains a list
    of `ModuleConfigs` representing sequential "blocks" in the model
    (e.g. a CNNBlock followed by an MLPBlock), initialize each block
    and combine them into a `nn.Sequential`.

    Args:
        experiment_config: the `ExperimentConfig` for this experiment.

    Returns:
        A `nn.Sequential` containing `nn.Module` objects matching the
            specification in the experiment config.
    """
    model_config = experiment_config.model
    modules = []
    for block_config in model_config:
        module_type = block_config.type
        module_params = block_config.params

        try:
            module_cls = MODELS[module_type.strip().lower()]
        except KeyError:
            raise ValueError(
                f"Unsupported module type {module_type}. "
                f"Current supported modules: {",".join(MODELS.keys())}"
            )

        try:
            module = module_cls(**module_params)
        except Exception as err:
            raise ValueError(
                f"Module {module_cls.__name__} could not be initialized "
                f"with parameters {module_params}: {err}"
            )
        modules.append(module)
    return nn.Sequential(*modules)
