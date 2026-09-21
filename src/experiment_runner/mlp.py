"""
A simple MLP-like `nn.Sequential` module.

Intended for use with MNISTFashion dataset, but the input dimensions,
linear layer size, and number of output classes are configurable.
"""

from torch import nn, Tensor


class MLPModule(nn.Module):
    """
    A simple `nn.Module` implementing a multi-layer perceptron (MLP)
    for use with the MNISTFashion dataset.
    """

    def __init__(self,
                 image_dim: int,
                 linear_layer_size: int,
                 output_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(
                in_features=image_dim**2,
                out_features=linear_layer_size,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=linear_layer_size,
                out_features=linear_layer_size,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=linear_layer_size,
                out_features=output_classes,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        flattened_input: Tensor = self.flatten(x)
        logits: Tensor = self.model(flattened_input)
        return logits
