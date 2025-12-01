import math

import torch
import torch.nn as nn

from src.alignment.alignment_factory import AlignmentFactory
from src.alignment.base_alignment_layer import BaseAlignmentLayer


@AlignmentFactory.register()
class MLPAlignmentLayer(BaseAlignmentLayer):

    def __init__(
        self,
        input_dim: int,
        dim_alignment: int,
        num_layers: int = 2,
        normalize_to_hypersphere: bool = False,
    ):
        super().__init__(input_dim=input_dim)
        self.normalize_to_hypersphere = normalize_to_hypersphere
        self.mlp = nn.Sequential(
            torch.nn.Linear(input_dim, dim_alignment),
        )
        for _ in range(num_layers - 1):
            self.mlp.append(nn.ReLU())
            self.mlp.append(torch.nn.Linear(dim_alignment, dim_alignment))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.mlp(z)
        # normalize logits to the hypersphere.
        if self.normalize_to_hypersphere:
            return nn.functional.normalize(z, p=2, dim=1)
        return z


def orthogonal_linear(layer: nn.Linear, gain: float = 1.0):
    """Orthogonal (or semi-orthogonal) weight init, zero bias."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


@AlignmentFactory.register()
class ResLowRankHead(BaseAlignmentLayer):
    r"""
    x ∈ ℝ^{in}  →  z ∈ ℝ^{out}

        z₀ = P  x                         # skip projection
        Δ  = W₂ · GELU( W₁ x )            # low-rank residual
        z  = z₀ + α · Dropout(Δ)          # learnable gate α∈[0,1]

    Args
    ----
    in_dim      : input dimension
    out_dim     : output / joint dimension
    rank        : bottleneck width  (default = out_dim // 4, min 8)
    dropout_p   : dropout on residual path (default 0.1)
    gate_init   : initial α (default 0 → residual starts off)
    """

    def __init__(
        self,
        input_dim: int,
        dim_alignment: int,
        rank: int | None = None,
        dropout_p: float = 0.1,
        gate_init: float = 0.0,
    ):
        super().__init__(input_dim=input_dim)
        rank = rank or max(8, dim_alignment // 4)

        # skip / projection
        self.P = nn.Linear(input_dim, dim_alignment, bias=False)
        orthogonal_linear(self.P)

        # low-rank residual path
        self.W1 = nn.Linear(input_dim, rank, bias=False)
        orthogonal_linear(self.W1)

        self.W2 = nn.Linear(rank, dim_alignment, bias=False)
        nn.init.zeros_(self.W2.weight)  # start near identity

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

        # learnable gate alpha, sigmoid(logit_alpha) -> alpha in [0,1]
        self.logit_alpha = nn.Parameter(torch.tensor(float("-inf")))
        if gate_init != 0.0:
            self.logit_alpha.data.fill_(math.log(gate_init / (1 - gate_init)))

    @property
    def alpha(self):
        return torch.sigmoid(self.logit_alpha)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z0 = self.P(z)
        delta = self.W2(self.dropout(self.act(self.W1(z))))
        return z0 + self.alpha * delta
