import torch


def get_optimizer_type(name: str = None, optimizer_name: str = None):
    if name is None:
        name = optimizer_name
    if name is None:
        raise ValueError("optimizer name is required")
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW
    if name == "adam":
        return torch.optim.Adam
    if name == "sgd":
        return torch.optim.SGD
    raise ValueError(f"Unknown optimizer type: {name}")
