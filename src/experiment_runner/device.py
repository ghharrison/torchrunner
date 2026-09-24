"""Utilities related to `torch.device`.

The device configuration is validated by creating a `torch.device`
and its availability is checked using `torch.empty`.
"""

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

    try:
        device = torch.device(device_name)
    except Exception as e:
        raise ValueError(
            f"Configured device {device_name} is not recognized "
            f"as a valid torch device: {e}"
        )

    try:
        torch.empty(0, device=device)
    except Exception as e:
        raise RuntimeError(
            f"Device {device_name} is a valid torch device, but is "
            f"not currently available: {e}"
        )

    return device
