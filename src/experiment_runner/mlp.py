from torch import nn, Tensor

MNIST_IMAGE_DIM: int = 28
MLP_LINEAR_SIZE: int = 512
MNIST_OUTPUT_CLASSES: int = 10


class MLPModule(nn.Module):
    """
    A simple `nn.Module` implementing a multi-layer perceptron (MLP)
    for use with the MNISTFashion dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten: nn.Module = nn.Flatten()  # flatten all dims
        self.model: nn.Module = nn.Sequential(
            nn.Linear(
                in_features=MNIST_IMAGE_DIM**2,
                out_features=MLP_LINEAR_SIZE,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=MLP_LINEAR_SIZE,
                out_features=MLP_LINEAR_SIZE,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=MLP_LINEAR_SIZE,
                out_features=MNIST_OUTPUT_CLASSES,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        flattened_input: Tensor = self.flatten(x)
        logits: Tensor = self.model(flattened_input)
        return logits
