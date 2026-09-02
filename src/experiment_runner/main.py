import torch
from torch import nn
from runner import Runner
from datasets.fashionmnist import create_loader
from mlp import MLPModule
from optimizer import create_optimizer
from experiment_config import ExperimentConfig

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
OPTIMIZER_TYPE = "sgd"

OPTIMIZER_CONFIG = {
    "type": OPTIMIZER_TYPE,
    "params": {
        "lr": LEARNING_RATE,
    },
}

CONFIG = {
    "batch_size": 64,
    "optimizer": OPTIMIZER_CONFIG,
}


def main():
    config = ExperimentConfig.from_dict(CONFIG)

    train_dataloader = create_loader(train=True, batch_size=config.batch_size)
    test_dataloader = create_loader(train=False, batch_size=config.batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # noqa
    model = MLPModule()
    runner = Runner(
        device=device,
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=create_optimizer(model, config.optimizer),
    )
    runner.train(train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
