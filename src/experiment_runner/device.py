from types import ModuleType

from experiment_runner.experiment_config import ExperimentConfig
import torch


def get_device(config: ExperimentConfig) -> torch.device:
    device_name = config.device.strip().lower()

    if device_name == 'cpu':
        return torch.device('cpu')

    if hasattr(torch.backends, device_name):
        backend: ModuleType = getattr(torch.backends, device_name)
        if hasattr(backend, 'is_available') and backend.is_available():
            return torch.device(device_name)
        print(
            f"Warning: configured device {device_name} is not available. "
            f"Falling back to CPU"
        )
        return torch.device('cpu')

    raise ValueError(
        f"Configured device {device_name} is not recognized "
        "as a valid torch backend."
    )
