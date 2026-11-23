"""Surrogate: a differentiable model of exact LPAP (FOUNDATION §2.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn

from lpap.geometry import PairGeometry
from lpap.transformer import TransformerBlock


@dataclass(frozen=True)
class SurrogateConfig:
    hidden_dim: int
    layer_count: int
    head_count: int
    rms_eps: float


class Surrogate(nn.Module):
    """Tokens ``(B, C, P)`` → local-class logits ``(B, C, k_max·P)``, class ``c = d·P + p``."""

    def __init__(self, geometry: PairGeometry, config: SurrogateConfig) -> None:
        super().__init__()
        if geometry.is_lossless:
            raise ValueError("the lossless pair has no surrogate")
        self.rms_eps = config.rms_eps
        self.input = nn.Linear(geometry.probe_count, config.hidden_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(config.hidden_dim, config.head_count) for _ in range(config.layer_count)
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, geometry.class_count)
        buckets = torch.arange(geometry.bucket_count)
        mask = (buckets[:, None] - buckets[None, :]) % geometry.bucket_count < geometry.k_max
        self.register_buffer("attention_mask", None if bool(mask.all()) else mask, persistent=False)

    def forward(self, tokens: Float[torch.Tensor, "batch c p"]) -> Float[torch.Tensor, "batch c classes"]:
        rms = tokens.square().mean(dim=(-2, -1), keepdim=True).add(self.rms_eps).sqrt()
        hidden = self.input(tokens / rms)
        for block in self.blocks:
            hidden = block(hidden, self.attention_mask)
        return self.output(self.output_norm(hidden))
