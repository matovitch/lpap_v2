"""Image grids for visual review: one labelled row per view, written as PNG (no imaging dependency)."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import torch
from jaxtyping import Float, UInt8

from lpap.data import hilbert_unflatten

ENERGY_DISPLAY_SCALE = 0.4


def image_tiles(values: Float[torch.Tensor, "batch 1 n"]) -> Float[torch.Tensor, "batch side side"]:
    """Model-order pixels in ``[0, 1]`` → square tiles (clamped)."""
    return hilbert_unflatten(values.detach().float().cpu()).squeeze(1).clamp(0.0, 1.0)


def energy_tiles(values: Float[torch.Tensor, "batch n"]) -> Float[torch.Tensor, "batch side side"]:
    """Signed energies → grey tiles, 0 at mid-grey, ``tanh`` squashed."""
    shown = 0.5 + 0.5 * torch.tanh(values.detach().float().cpu() / ENERGY_DISPLAY_SCALE)
    return hilbert_unflatten(shown.unsqueeze(1)).squeeze(1)


def grid(
    rows: list[Float[torch.Tensor, "batch side side"]], *, scale: int = 2, gap: int = 2
) -> UInt8[torch.Tensor, "height width"]:
    """Rows of equally sized tiles in ``[0, 1]`` → one uint8 image, white gaps."""
    columns = max(row.shape[0] for row in rows)
    side = rows[0].shape[-1] * scale
    height = len(rows) * side + (len(rows) + 1) * gap
    width = columns * side + (columns + 1) * gap
    canvas = torch.ones(height, width)
    for r, row in enumerate(rows):
        tiles = row.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
        top = gap + r * (side + gap)
        for c, tile in enumerate(tiles):
            left = gap + c * (side + gap)
            canvas[top : top + side, left : left + side] = tile
    return (canvas * 255.0).round().to(torch.uint8)


def write_png(path: str | Path, pixels: UInt8[torch.Tensor, "height width"]) -> Path:
    """8-bit greyscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = pixels.shape
    raw = b"".join(b"\x00" + bytes(row) for row in pixels.contiguous().numpy())

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )
    tmp.replace(path)
    return path
