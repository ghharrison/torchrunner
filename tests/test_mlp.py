import torch

from experiment_runner.model.mlp import MLPModule


FASHION_IMAGE_DIM = 28
LINEAR_LAYER_SIZE = 512
N_CLASSES = 10


def test_mlp_module() -> None:
    mlp = MLPModule(
        image_dim=FASHION_IMAGE_DIM,
        linear_layer_size=LINEAR_LAYER_SIZE,
        output_classes=N_CLASSES,
    )

    # random batch of 32 tensors of size (1, 28, 28)
    X = torch.randn(32, 1, FASHION_IMAGE_DIM, FASHION_IMAGE_DIM)
    prediction = mlp(X)
    assert prediction.shape == (32, N_CLASSES)
