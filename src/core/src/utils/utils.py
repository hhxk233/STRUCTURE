import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from loguru import logger


def fix_random_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0
    log_messages: bool = False

    def __post_init__(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            return
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.log_messages:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float = 1.0):
    if isinstance(parameters, torch.nn.Module):
        parameters = parameters.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def save_checkpoint(run_dir: Path, save_dict: dict, epoch: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(save_dict, ckpt_path)
    last_path = run_dir / "checkpoint_last.pt"
    torch.save(save_dict, last_path)
    logger.debug(f"Saved checkpoint to {ckpt_path}")
