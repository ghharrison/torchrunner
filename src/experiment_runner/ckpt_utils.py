import torch


def save_checkpoint(
        model: torch.nn.Module,
        checkpoint_dir: str,
        modeltype: str,
        epoch: int,
):
    torch.save(
        model.state_dict(),
        checkpoint_dir / f'{modeltype}_weights_{epoch}.pth'
    )
