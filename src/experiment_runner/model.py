from typing import Mapping

from torch import nn

import experiment_runner.mlp as mlp
from experiment_runner.experiment_config import ExperimentConfig

MODELS: Mapping[str, type[nn.Module]] = {
    "mlp": mlp.MLPModule,
}


def get_module(
        experiment_config: ExperimentConfig) -> nn.Module:
    model_config = experiment_config.model
    model_type = model_config.type
    model_params = model_config.params

    try:
        module_cls = MODELS[model_type.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported model type {model_type}. "
            f"Current supported models: {",".join(MODELS.keys())}"
        )

    try:
        model = module_cls(**model_params)
    except TypeError as err:
        raise ValueError(
            f"Module {module_cls.__name__} could not be initialized "
            f"with parameters {model_params}: {err}"
        )
    return model
