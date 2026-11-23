"""Stage 2: encode every image once with the stage-1 flow (FOUNDATION §5).

The bank is never written to disk: molab only persists small source files, and
it is a deterministic function of ``flow/best.pt`` and the image set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from lpap.data import EnergyBank, ImageSet
from lpap.flow import encode, render
from lpap.runtime.config import ModelConfig
from lpap.runtime.hub import ArtifactStore
from lpap.stages.flow import load_field, top_share


@dataclass(frozen=True)
class BankStageConfig:
    steps: int
    batch_size: int
    compile: bool
    check_count: int


@torch.no_grad()
def build(
    model: ModelConfig,
    config: BankStageConfig,
    images: ImageSet,
    store: ArtifactStore,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[EnergyBank, dict[str, float]]:
    """Bank of fp32-integrated energies (stored float16) and summary statistics."""
    eager = load_field(model, store, images.device).eval()
    field = torch.compile(eager, dynamic=False) if config.compile else eager
    count = images.images.shape[0]
    energies = torch.empty(count, images.images[0].numel(), dtype=torch.float16, device=images.device)
    for start in range(0, count, config.batch_size):
        indices = torch.arange(start, min(start + config.batch_size, count), device=images.device)
        # The last chunk has another shape; run it eagerly instead of recompiling.
        chunk_field = field if indices.numel() == config.batch_size else eager
        energies[indices] = encode(chunk_field, images.batch(indices), config.steps).half()
        if progress is not None:
            progress(int(indices[-1]) + 1, count)
    bank = EnergyBank(energies, images.device)

    check = images.validation[: config.check_count]
    energy = bank.batch(check)
    stats = {
        "count": float(count),
        "energy_mean": float(energy.mean()),
        "energy_std": float(energy.std()),
        "energy_top10": float(top_share(energy, 0.1)),
        "energy_abs_max": float(bank.energies.abs().max()),
        # From the stored float16 rows, so this includes the storage rounding.
        "round_trip_l2": float((render(eager, energy, config.steps) - images.batch(check)).square().mean()),
    }
    return bank, stats
