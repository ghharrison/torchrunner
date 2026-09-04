import yaml
from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class OptimizerConfig:
    type: str = "sgd"
    params: Mapping = field(default_factory=dict)

    def keys(self):
        return ['type', 'params']

    def __getitem__(self, key):
        return getattr(self, key)


@dataclass
class ExperimentConfig:
    experiment_name: str = "default"
    batch_size: int = 64
    epochs: int = 10
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def keys(self):
        return ['experiment_name', 'batch_size', 'epochs', 'optimizer']

    def __getitem__(self, key):
        return dict(self.optimizer) if key == 'optimizer' else getattr(self, key)  # noqa

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
            config = yaml.safe_load(file)
        return cls.from_dict(config)
