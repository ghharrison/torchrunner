"""
A simple convolutional neural network `nn.Sequential` module.

Intended for use with CIFAR-10 dataset, but the input dimensions,
kernel size, stride, padding, and number of output classes
are configurable.
"""

from torch import nn, Tensor


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class CNNModule(nn.Module):
    """
    A simple `nn.Module` implementing a convolutional neural network (CNN)
    for use with the CIFAR-10 dataset.
    """

    def __init__(self,
                 image_dim: int,
                 in_channels: int,
                 conv_layer_size: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_classes: int) -> None:
        super().__init__()
        self.preprocess = Lambda(
            lambda x: x.view(-1, in_channels, image_dim, image_dim))
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                conv_layer_size,
                kernel_size,
                stride,
                padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_layer_size,
                conv_layer_size,
                kernel_size,
                stride,
                padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_layer_size,
                output_classes,
                kernel_size,
                stride,
                padding,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
        )
        self.postprocess = Lambda(lambda x: x.view(x.size(0), -1))

    def forward(self, x: Tensor) -> Tensor:
        x: Tensor = self.preprocess(x)
        logits: Tensor = self.postprocess(self.model(x))
        return logits
