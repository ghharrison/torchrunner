from types import ModuleType

import torch


def get_device(device_name: str) -> torch.Device:
    device_name: str = device_name.strip().lower()

    if device_name == 'cpu':
        return torch.device('cpu')

    if hasattr(torch.backends, device_name):
        backend: ModuleType = getattr(torch.backends, device_name)
        if hasattr(backend, 'is_available') and backend.is_available():
            return torch.device(device_name)

    print(f"Warning: configured device {device_name} is not available."
          " Falling back to CPU")
    return torch.device('cpu')
