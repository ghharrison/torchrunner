"""
A simple convolutional neural network `nn.Sequential` module.

Intended for use with CIFAR-10 dataset, but the input dimensions,
kernel size, stride, padding, and number of output classes
are configurable.
"""

from torch import nn, Tensor

POOLING_MODES = [
    "max",
    "avg",
]


def compute_output_dim(
        input_dim: int,
        kernel_size: int,
        padding: int | str = 0,
        stride: int = 1,
        dilation: int = 1,
) -> int:
    if type(padding) is str:
        if padding == "valid":
            padding = 0
        elif padding == "same":
            return input_dim
        else:
            if type(padding) is not int:
                raise ValueError(
                    f"invalid padding value {padding} - must be an integer, "
                    "'valid', or 'same'"
                )
    out_dim = int(
        (input_dim + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
    )
    return out_dim


class CNNBlock(nn.Module):
    """
    A simple `nn.Module` implementing a convolutional neural network (CNN)
    for use with the CIFAR-10 dataset. All convolutions are done with
    `nn.Conv2d`.

    This does NOT automatically determine output shapes after each layer;
    you must configure the block dimensions correctly.

    Current assumptions:
    - input is batched into a shape of [B, C, H, W].
    - images are square (`input_image_dim x input_image_dim`).
    - The same padding is used in horizontal and vertical dimensions.
    - Stride and dilation are 1.
    - ReLU activation is used after every convolution.
    - Pooling is done after every convolution layer, or skipped if
        `pool_factor` is None.
    - Pooling is done with `nn.MaxPool2d` or `nn.AvgPool2d`.
    """
    def __init__(self,
                 input_image_dim: int,
                 in_channels: int,
                 out_channels: list[int],
                 kernel_sizes: list[int],
                 padding: int | str = 0,
                 pool_factor: int | None = None,
                 pool_mode: str = "max") -> None:
        super().__init__()

        layers = []
        input_dim = input_image_dim
        dilation = 1
        stride = 1

        for i in range(len(out_channels)):
            output_dim = compute_output_dim(
                input_dim,
                kernel_sizes[i],
                padding,
                stride,
                dilation,
            )
            if output_dim <= 0:
                raise ValueError(
                    "Invalid Conv2d configuration: using a convolutional "
                    f"layer on a {input_dim} x {input_dim} image with "
                    f"kernel size {kernel_sizes[i]}, padding {padding}, "
                    f"dilation {dilation}, and stride {stride} would produce "
                    f"output image of dimension {output_dim} x {output_dim}"
                )
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                )
            )
            # Update for next loop iteration
            input_dim = output_dim
            in_channels = out_channels[i]

            layers.append(nn.ReLU())
            if pool_factor:
                if pool_factor > 1 and output_dim <= 1:
                    raise ValueError(
                        f"Using a pooling factor of {pool_factor} when the "
                        f"preceding Conv2d layer produces a "
                        f"{output_dim}x{output_dim} image would produce a "
                        f"0 x 0 image"
                    )
                match pool_mode:
                    case "max":
                        pool_cls = nn.MaxPool2d
                    case "avg":
                        pool_cls = nn.AvgPool2d
                    case _:
                        raise ValueError(
                            f"Configured pooling mode {pool_mode} is not a "
                            f"valid pooling type; options are ['max', 'avg']"
                        )
                layers.append(
                    pool_cls(pool_factor, pool_factor)
                )

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
