"""
A simple MLP-like `nn.Sequential` module.

Intended for use with MNISTFashion dataset, but the input dimensions,
linear layer size, and number of output classes are configurable.
"""

from torch import nn, Tensor


class MLPBlock(nn.Module):
    """
    A simple `nn.Module` implementing a multi-layer perceptron (MLP)
    for use with the MNISTFashion dataset.
    """

    def __init__(self,
                 in_channels: int,
                 input_image_dim: int,
                 hidden_layer_sizes: list[int],
                 output_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        in_features = in_channels*(input_image_dim**2)
        for i in range(len(hidden_layer_sizes)):
            layers.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=hidden_layer_sizes[i],
                )
            )
            # update for next loop iteration
            in_features = hidden_layer_sizes[i]
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(
                in_features=hidden_layer_sizes[-1],
                out_features=output_classes,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        flattened_input: Tensor = self.flatten(x)
        logits: Tensor = self.model(flattened_input)
        return logits
