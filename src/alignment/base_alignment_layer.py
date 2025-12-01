from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseAlignmentLayer(ABC, nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
