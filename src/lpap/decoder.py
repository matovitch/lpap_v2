"""Decoder: infers the probe of every bucket from the whole code (FOUNDATION §2.6)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn

from lpap.geometry import PairGeometry
from lpap.transformer import TransformerBlock


@dataclass(frozen=True)
class DecoderConfig:
    hidden_dim: int
    layer_count: int
    head_count: int


class Decoder(nn.Module):
    """``(a, w)`` → probe logits ``(B, C, P)`` over the collision set ``S(b, d_b)``.

    Never receives probe information. Buckets fed from the same source lane
    share the lane embedding, so they can coordinate on distinct probes.
    """

    def __init__(self, geometry: PairGeometry, config: DecoderConfig) -> None:
        super().__init__()
        if geometry.is_lossless:
            raise ValueError("the lossless pair has no decoder")
        hidden = config.hidden_dim
        self.amplitude = nn.Linear(1, hidden)
        self.source_lane = nn.Embedding(geometry.bucket_count, hidden)
        self.dib = nn.Linear(geometry.k_max, hidden, bias=False)
        self.position = nn.Parameter(torch.randn(geometry.bucket_count, hidden) * 0.02)
        self.blocks = nn.ModuleList(TransformerBlock(hidden, config.head_count) for _ in range(config.layer_count))
        self.output_norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, geometry.probe_count)
        self.register_buffer("source_lanes", geometry.source_lanes().cpu(), persistent=False)

    def forward(
        self,
        amplitudes: Float[torch.Tensor, "batch c"],
        dib_weights: Float[torch.Tensor, "batch c k"],
    ) -> Float[torch.Tensor, "batch c p"]:
        lanes = self.source_lane(self.source_lanes)
        hidden = (
            self.amplitude(amplitudes.unsqueeze(-1))
            + torch.einsum("bck,ckh->bch", dib_weights.to(lanes.dtype), lanes)
            + self.dib(dib_weights.to(self.dib.weight.dtype))
            + self.position
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden))
