# Torchrunner: A PyTorch experiment-running tool

A Python package for running configurable PyTorch experiments. 

This project currently supports image-classification experiments
using PyTorch and a defined set of dataset/model/optimizer implementations.
The core training and evaluation components are deliberately independent
of individual model implementations.

## Features
- Persists experiment state, including model/optimizer/loss function state and
RNG state, to allow tracking experiment progress and resuming from
interrupted training runs. 
- Supports a variety of model architectures, optimizers, and loss functions.
- Experiment configs allow easy experimentation across multiple configurations
without requiring changes to source.

## Installation
Navigate to the directory containing the checked-out repo and use `pip install` 
to create an editable install. Dependencies such as `torch` will be installed
automatically.
```
pip install -e .
```

## Usage
Running an experiment requires an experiment config, which defines key aspects
of the experiment such as the choice of architecture and optimizer. An example
config is located at `conf/default.yaml`. 
```
experiment-runner --config path/to/config.yaml
```
Alternatively, providing a path to a Torchrunner-created checkpoint with 
`--checkpoint` attempts to resume training from the last uninterrupted epoch:
```
experiment-runner --checkpoint experiments/foo/20XX-MM-DD_HH-MM-SS/checkpoints/epoch_03.pt
```