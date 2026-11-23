"""Pre-norm transformer block with circular RoPE over the bucket ring."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def circular_rope_tables(
    token_count: int, rotary_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frequencies ``2π m / T`` for ``m = 1..rotary_dim/2``: a shift by ``T`` is the identity."""
    positions = torch.arange(token_count, device=device, dtype=torch.float32)
    harmonics = torch.arange(1, rotary_dim // 2 + 1, device=device, dtype=torch.float32)
    angles = positions[:, None] * (2.0 * math.pi / token_count) * harmonics[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(tensor: torch.Tensor, cosines: torch.Tensor, sines: torch.Tensor) -> torch.Tensor:
    rotary_dim = 2 * cosines.shape[-1]
    rotary, rest = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    rotated = torch.stack((even * cosines - odd * sines, even * sines + odd * cosines), dim=-1).flatten(-2)
    return torch.cat((rotated, rest), dim=-1)


class CircularSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_count: int) -> None:
        super().__init__()
        if hidden_dim % head_count:
            raise ValueError("hidden_dim must be divisible by head_count")
        self.head_count = head_count
        self.head_dim = hidden_dim // head_count
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        batch, count, hidden = tokens.shape
        query, key, value = (
            part.reshape(batch, count, self.head_count, self.head_dim).transpose(1, 2)
            for part in self.qkv(tokens).chunk(3, dim=-1)
        )
        rotary_dim = self.head_dim - self.head_dim % 2
        if rotary_dim:
            cosines, sines = circular_rope_tables(count, rotary_dim, tokens.device, query.dtype)
            query, key = apply_rope(query, cosines, sines), apply_rope(key, cosines, sines)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        return self.output(attended.transpose(1, 2).reshape(batch, count, hidden))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, head_count: int, mlp_multiplier: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = CircularSelfAttention(hidden_dim, head_count)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_multiplier * hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_multiplier * hidden_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``attention_mask``: bool ``(T, T)``, ``True`` = may attend."""
        tokens = tokens + self.attention(self.attention_norm(tokens), attention_mask)
        return tokens + self.mlp(self.mlp_norm(tokens))
