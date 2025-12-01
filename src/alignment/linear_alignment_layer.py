import torch
import torch.nn as nn

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer


@AlignmentFactory.register()
class LinearAlignmentLayer(BaseAlignmentLayer):

    def __init__(
        self,
        input_dim: int,
        dim_alignment: int,
        normalize_to_hypersphere: bool = False,
    ):
        super().__init__(input_dim=input_dim)
        self.normalize_to_hypersphere = normalize_to_hypersphere
        self.linear_mapping = torch.nn.Linear(input_dim, dim_alignment)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.linear_mapping(z)
        # normalize logits to the hypersphere.
        if self.normalize_to_hypersphere:
            return nn.functional.normalize(z, p=2, dim=1)
        return z
