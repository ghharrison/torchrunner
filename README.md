# Torchrunner: A PyTorch experiment-running tool

## Overview

A configurable, modular framework for running and comparing PyTorch image-classification experiments.

The framework separates experiment management, training, and evaluation from individual model implementations. Experiments are defined through configuration files, allowing users to vary datasets, data transformations, model architectures, and training settings without rewriting the core workflow.

This makes it straightforward to investigate questions such as:

- **Data transformations:** How do different preprocessing and augmentation transforms affect classification performance?
- **Model architecture:** How does performance change when varying architectural choices, such as the number of MLP hidden layers or the kernel size of CNN layers?
- **Ablation studies:** How much does a particular model component contribute to performance—for example, what happens when a CNN block is removed from a trained CNN–MLP classifier?

## Features
- Persists experiment state, including model/optimizer/loss function state and
RNG state, to allow tracking experiment progress and resuming from
interrupted training runs. 
- Supports a variety of model architectures, optimizers, and loss functions.
- Experiment configs allow easy experimentation across multiple configurations
without requiring changes to source.
- Supports multi-layer perceptron (MLP) and convolutional neural network (CNN) blocks.
- Configurable choice of datasets including FashionMNIST and CIFAR-10.
- Improve reproducibility of experiments through recording and controlling RNG state across Python, NumPy, PyTorch, and CUDA.

## Limitations
- This framework is currently set up to perform image classification experiments on vision datasets. Datasets configured in the experiment config are expected to be under `torchvision.datasets` and any transforms are expected to come from `torchvision.transforms.v2`. 
- Experiments are executed locally on a single machine. Distributed and remote execution are considered out-of-scope.
- While reproducibility is significantly improved by controlling RNG state, certain operations may introduce non-determinism. Perfect reproducibility across all cases is not guaranteed.

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

## Writing experiment configs

Experiment configs, written in YAML format, define the experiment Torchrunner will execute. 

### Required fields

The following fields are required regardless of the choice of model architecture, as shown in the example `conf/default.yaml`:
```
experiments_root: experiments
experiment_name: default
batch_size: 64
epochs: 10
device: cpu
```
- `experiments_root`: The root directory where all experiments are logged to, relative to the working directory. 
- `experiment_name`: A name to identify the experiment. Each experiment run will be logged as a subdirectory of this experiment directory.
- `batch_size`: The batch size used by `DataLoader` instances for both training and evaluation.
- `epochs`: The number of epochs to run the experiment for. Each epoch includes a training step and an evaluation step.
- `device`: The `torch.device` to run on. Commonly either `cpu` or `cuda`.

### Dataset fields
The `dataset` field contains settings related to the dataset the experiment is run against. Currently, this assumes that the configured dataset is findable under `torchvision.datasets` and any transforms used are expected to be under `torchvision.transforms.v2`. Again using `conf/default.yaml` as our example:

```
dataset:
  name: FashionMNIST
  transforms:
    train:
      - name: ToImage
        params: {}
      - name: ToDtype
        params:
          dtype: float32
          scale: true
    eval:
      - name: ToImage
        params: {}
      - name: ToDtype
        params:
          dtype: float32
          scale: true
```
- `name`: The name of the dataset. This will be used to import the dataset from `torchvision.datasets`. 
- `transforms`: The lists of transforms that will be applied in the `DataLoader` to each image from the dataset during training and evaluation. Transforms must be specified for both `train` and `eval` as some transforms (such as `RandomCrop`) are generally only applied during training. Any transform in the list is expected to be under `torchvision.transforms.v2`.

A more complete example appears under `fashion.yaml`. Here we can see `RandomHorizontalFlip`, `RandomVerticalFlip`, and `RandomCrop` are applied only during training, while `ToImage`, `ToDtype`, and `Normalize` are applied for both training and evaluation:

```
dataset:
  name: FashionMNIST
  transforms:
    train:
      - name: RandomHorizontalFlip
        params: {}
      - name: RandomVerticalFlip
        params: {}
      - name: RandomCrop
        params:
          size: 28
          padding: 2
      - name: ToImage
        params: {}
      - name: ToDtype
        params:
          dtype: float32
          scale: true
      - name: Normalize
        params:
          mean:
            - 0.2862
          std:
            - 0.3531
    eval:
      - name: ToImage
        params: {}
      - name: ToDtype
        params:
          dtype: float32
          scale: true
      - name: Normalize
        params:
          mean:
            - 0.2862
          std:
            - 0.3531
```

### Model fields
The `model` field defines the architecture and hyperparameters of the model under experimentation. The modules are expressed as a list, allowing combination of different module "blocks" into a larger model, such as a CNN + MLP architecture. Using `conf/cifar.yaml` as an example, we define a model of a CNN block plus MLP block that takes 3 input channels and ultimately produces 10 output classes:
```
model:
  - type: cnn
    params:
      in_channels: 3
      pool_factor: 2 
      out_channels: [16, 32, 64]
      input_image_dim: 32
      kernel_sizes: [5, 5, 5]
      padding: 'same'
  - type: mlp
    params:
      in_channels: 64
      input_image_dim: 4
      hidden_layer_sizes: [512, 256, 128]
      output_classes: 10
```
Torchrunner does not automatically compute dimensions of each layer's output or correct mismatched dimensions; the user must specify these correctly themselves.

Each item under `model` is expected to have a `type` field identifying the module type and `params`, which is a generic set of keyword arguments that will be particular to each type of module. Additional types of module can be supported by adding their implementations under `src\experiment_runner\model`. 

### Other fields

An experiment config also requires a `loss_fn` and `optimizer` field to configure the loss function and optimizer. These both follow the same format:
- A `type` field specifying the name of the loss function or optimizer (`loss.py` and `optimizer.py` contain mappings of loss/optimizer names to class definitions)
- A `params` mapping of keyword arguments to be passed to the constructor

`default.py` contains an example using `nn.CrossEntropyLoss` loss and `torch.optim.SGD` optimizer:
```
loss_fn:
  type: crossentropy
  params: {}
optimizer:
  type: sgd
  params:
    lr: 0.001
```