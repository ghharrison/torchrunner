from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml


@dataclass
class ModuleConfig:
    type: str
    params: Mapping = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiments_root: str = "experiments"
    experiment_name: str = "default"
    batch_size: int = 64
    epochs: int = 10
    model: ModuleConfig
    loss_fn: ModuleConfig
    optimizer: ModuleConfig
    device: str = "cpu"

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, Any]) -> "ExperimentConfig":
        config_dict['model'] = ModuleConfig(**config_dict.get("model", {}))
        config_dict['loss_fn'] = ModuleConfig(**config_dict.get("loss_fn", {}))
        config_dict['optimizer'] = ModuleConfig(**config_dict.get("optimizer", {}))  # noqa

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentConfig":
        with open(config_path, 'r') as file:
            config: Mapping[str, Any] = yaml.safe_load(file)
            return cls.from_dict(config)
