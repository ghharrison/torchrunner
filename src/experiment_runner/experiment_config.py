from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml


@dataclass
class OptimizerConfig:
    type: str = "sgd"
    params: Mapping = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment_name: str = "default"
    batch_size: int = 64
    epochs: int = 10
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    device: str = "cpu"

    @classmethod
    def from_dict(cls, cfg: dict) -> "ExperimentConfig":
        return cls(
            experiment_name=cfg.get('experiment_name'),
            batch_size=cfg.get("batch_size", 64),
            epochs=cfg.get("epochs", 10),
            optimizer=OptimizerConfig(**cfg.get("optimizer", {})),
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentConfig":
        with open(config_path, 'r') as file:
            config: Mapping[str, Any] = yaml.safe_load(file)
        return cls.from_dict(config)
