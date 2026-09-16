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
    model: ModuleConfig = field(default_factory=ModuleConfig)
    loss_fn: ModuleConfig = field(default_factory=ModuleConfig)
    optimizer: ModuleConfig = field(default_factory=ModuleConfig)
    device: str = "cpu"

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentConfig":
        with open(config_path, 'r') as file:
            config: Mapping[str, Any] = yaml.safe_load(file)
        config['model'] = ModuleConfig(**config.get("model", {}))
        config['loss_fn'] = ModuleConfig(**config.get("loss_fn", {}))
        config['optimizer'] = ModuleConfig(**config.get("optimizer", {}))

        return cls(**config)
