from torch import nn

MNIST_IMAGE_DIM = 28
MLP_LINEAR_SIZE = 512
MNIST_OUTPUT_CLASSES = 10


class MLPModule(nn.Module):
    """
    A simple `nn.Module` implementing a multi-layer perceptron (MLP)
    for use with the MNISTFashion dataset.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # flatten all dims
        self.model = nn.Sequential(
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

    def forward(self, x):
        flattened_input = self.flatten(x)
        logits = self.model(flattened_input)
        return logits
