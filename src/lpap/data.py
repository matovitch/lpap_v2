"""Images: Hilbert ordering, archive fetch, deterministic train/validation split."""

from __future__ import annotations

from pathlib import Path

import torch
from jaxtyping import Float, Int, UInt8

IMAGES_BUCKET = "matovitch/lpap-images"
IMAGES_REMOTE = "images_32x32_gray.pt.zst"
IMAGES_LOCAL = "data/images_32x32_gray.pt"
SPLIT_SEED = 20261123
VALIDATION_FRACTION = 0.05


def hilbert_order(side: int) -> Int[torch.Tensor, n]:
    """``order[i]`` is the row-major pixel index of Hilbert position ``i``."""
    if side <= 0 or side & (side - 1):
        raise ValueError("side must be a positive power of two")
    order = []
    for distance in range(side * side):
        x = y = 0
        rest = distance
        scale = 1
        while scale < side:
            rx = 1 & (rest // 2)
            ry = 1 & (rest ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = scale - 1 - x, scale - 1 - y
                x, y = y, x
            x += scale * rx
            y += scale * ry
            rest //= 4
            scale *= 2
        order.append(y * side + x)
    return torch.tensor(order, dtype=torch.long)


def hilbert_flatten(images: Float[torch.Tensor, "batch 1 side side"]) -> Float[torch.Tensor, "batch 1 n"]:
    side = images.shape[-1]
    return images.flatten(2)[..., hilbert_order(side).to(images.device)]


def hilbert_unflatten(sequence: Float[torch.Tensor, "batch 1 n"]) -> Float[torch.Tensor, "batch 1 side side"]:
    side = int(round(sequence.shape[-1] ** 0.5))
    order = hilbert_order(side).to(sequence.device)
    out = torch.empty_like(sequence)
    out[..., order] = sequence
    return out.reshape(*sequence.shape[:-1], side, side)


def ensure_images(root: str | Path = ".") -> Path:
    """Return the local image archive, downloading and decompressing if missing."""
    root = Path(root)
    pt_path = root / IMAGES_LOCAL
    if pt_path.is_file():
        return pt_path
    import zstandard
    from huggingface_hub import HfFileSystem

    pt_path.parent.mkdir(parents=True, exist_ok=True)
    zst_path = pt_path.with_suffix(".pt.zst")
    if not zst_path.is_file():
        partial = zst_path.with_suffix(".partial")
        HfFileSystem(token=False).get(f"hf://buckets/{IMAGES_BUCKET}/{IMAGES_REMOTE}", str(partial))
        partial.replace(zst_path)
    tmp = pt_path.with_suffix(".tmp")
    with zst_path.open("rb") as source, tmp.open("wb") as target:
        zstandard.ZstdDecompressor().copy_stream(source, target)
    tmp.replace(pt_path)
    return pt_path


def load_images(path: str | Path) -> UInt8[torch.Tensor, "count 1 side side"]:
    images = torch.load(Path(path), map_location="cpu", mmap=True, weights_only=True)["images"]
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("expected uint8 images of shape (count, 1, side, side)")
    return images


def split_indices(count: int) -> tuple[Int[torch.Tensor, train], Int[torch.Tensor, validation]]:
    """Fixed 95/5 split, a pure function of ``count`` (FOUNDATION §6)."""
    order = torch.randperm(count, generator=torch.Generator().manual_seed(SPLIT_SEED))
    validation_count = max(1, round(count * VALIDATION_FRACTION))
    return order[validation_count:].sort().values, order[:validation_count].sort().values


def to_model_input(images: UInt8[torch.Tensor, "batch 1 side side"]) -> Float[torch.Tensor, "batch 1 n"]:
    """``/255`` then Hilbert order."""
    return hilbert_flatten(images.to(torch.float32) / 255.0)
