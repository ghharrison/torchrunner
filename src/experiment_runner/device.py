"""Utilities related to `torch.device`.

The set of supported devices is inferred from the attributes of
`torch.backends`.
"""

from types import ModuleType

from experiment_runner.experiment_config import ExperimentConfig
import torch


def get_device(config: ExperimentConfig) -> torch.device:
    """
    Read the configured device from an `ExperimentConfig` and return a
    `torch.device` representing it. If the device is unavailable or is
    not a supported backend, falls back to CPU.

    Args:
        config: An `ExperimentConfig` for the current experiment run.

    Returns:
        A `torch.device` instance.
    """
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
