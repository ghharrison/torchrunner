# Project: mini experiment runner

Taking a trivial toy example from the PyTorch tutorial, gradually build it up into a working example of experiment software. Goals include
- Modularize the code, logically placing different functionality in corresponding files under `src/`
- Make the code configurable using dataclasses, JSON/YAML configs and command-line arguments
- Generate experiment directories for each run, storing metrics, logs, and checkpoints
- Extend checkpoint saving to include state of training process, allowing interrupted training runs to be resumed
- Export metrics to JSON or CSV in addition to CLI output. Add utilities to plot metrics.
- Unit test key components
- Longer-term: use a larger dataset (CIFAR-10) and a more advanced network (CNN)
- Longer-term: integrate with experiment-running tools like WnB