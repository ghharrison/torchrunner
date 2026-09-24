"""
Dataclass for experiment configs.

Also includes a generic Pydantic model intended to configure
`nn.Module`s, including models, optimizers, or some loss functions.
"""
from typing import Any, Self

from pydantic import BaseModel, Field
import yaml


class ModuleConfig(BaseModel):
    type: str
    params: dict[str, Any]


class TransformConfig(BaseModel):
    name: str
    params: dict[str, Any]


class DatasetTransformConfig(BaseModel):
    train: list[TransformConfig] = Field(default_factory=list)
    eval: list[TransformConfig] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    name: str
    root: str = "data"
    transforms: DatasetTransformConfig


class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    model: ModuleConfig
    loss_fn: ModuleConfig
    optimizer: ModuleConfig
    experiments_root: str = "experiments"
    experiment_name: str = "default"
    batch_size: int = 64
    epochs: int = 10
    device: str = "cpu"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """
        Create an `ExperimentConfig` object from a dictionary
        representing its fields.

        Args:
            config_dict: A `dict[str, Any]` representing the config.
            The fields `dataset`, `model`, `loss_fn`, and `optimizer`
            must be present in the config; others have default values.

        Returns:
            An `ExperimentConfig` whose fields match the input dictionary.
        """
        config_dict['model'] = ModuleConfig(
            **config_dict.get("model", {}),
        )
        config_dict['loss_fn'] = ModuleConfig(
            **config_dict.get("loss_fn", {}),
        )
        config_dict['optimizer'] = ModuleConfig(
            **config_dict.get("optimizer", {}),
        )
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, config_path: str) -> Self:
        """
        Create an `ExperimentConfig` from a YAML config
        at the given path.

        Args:
            config_path: A string of the path to the YAML config.

        Returns:
            An `ExperimentConfig` matching the contents of the YAML.
        """
        with open(config_path, 'r') as file:
            config: dict[str, Any] = yaml.safe_load(file)
            return cls.from_dict(config)
