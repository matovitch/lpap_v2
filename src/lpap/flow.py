"""Bidirectional flow field and its integration (FOUNDATION §2.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FlowConfig:
    width: int
    time_dim: int
    time_scale: float
    dilation_cycles: int
    dilations: tuple[int, ...]
    kernel_size: int


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time: Float[torch.Tensor, batch]) -> Float[torch.Tensor, "batch dim"]:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype) * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = self.scale * time[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class DilatedResidualBlock(nn.Module):
    def __init__(self, *, width: int, time_dim: int, dilation: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.film = nn.Linear(time_dim, 2 * width)
        self.conv = nn.Conv1d(width, width, kernel_size, padding=dilation * (kernel_size // 2), dilation=dilation)

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(time_embedding).chunk(2, dim=-1)
        inner = self.norm(hidden) * (1.0 + scale[:, :, None]) + shift[:, :, None]
        return hidden + self.conv(F.silu(inner))


class FlowField(nn.Module):
    """Velocity ``v(x, t)`` on ``t ∈ [-1, 1]``: image → energy → image."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        if config.time_dim % 2:
            raise ValueError("time_dim must be even")
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(config.time_dim, config.time_scale),
            nn.Linear(config.time_dim, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.input = nn.Conv1d(1, config.width, 1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(
                width=config.width, time_dim=config.time_dim, dilation=dilation, kernel_size=config.kernel_size
            )
            for _ in range(config.dilation_cycles)
            for dilation in config.dilations
        )
        self.output_norm = nn.GroupNorm(1, config.width)
        self.output = nn.Conv1d(config.width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, values: Float[torch.Tensor, "batch 1 n"], time: Float[torch.Tensor, batch]
    ) -> Float[torch.Tensor, "batch 1 n"]:
        time_embedding = self.time_embedding(time.to(values.dtype))
        hidden = self.input(values)
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        return self.output(F.silu(self.output_norm(hidden)))


def integrate(
    field: nn.Module,
    start: Float[torch.Tensor, "batch 1 n"],
    *,
    t0: float,
    t1: float,
    steps: int,
) -> Float[torch.Tensor, "batch 1 n"]:
    """Euler midpoint, always fp32 (autocast disabled)."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    delta = (t1 - t0) / steps
    values = start.float()
    with torch.autocast(device_type=values.device.type, enabled=False):
        for step in range(steps):
            time = torch.full((values.shape[0],), t0 + (step + 0.5) * delta, device=values.device)
            values = values + delta * field(values, time)
    return values


def encode(field: nn.Module, image: Float[torch.Tensor, "batch 1 n"], steps: int) -> Float[torch.Tensor, "batch n"]:
    return integrate(field, image, t0=-1.0, t1=0.0, steps=steps).squeeze(1)


def render(field: nn.Module, energy: Float[torch.Tensor, "batch n"], steps: int) -> Float[torch.Tensor, "batch 1 n"]:
    return integrate(field, energy.unsqueeze(1), t0=0.0, t1=1.0, steps=steps)


def sample_prior(
    shape: tuple[int, ...],
    *,
    sigma: float,
    scale: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Signed log-normal: ``±scale · exp(sigma · Z)``, fair sign."""
    magnitude = scale * torch.exp(sigma * torch.randn(shape, generator=generator, device=device))
    sign = torch.randint(0, 2, shape, generator=generator, device=device) * 2 - 1
    return magnitude * sign
