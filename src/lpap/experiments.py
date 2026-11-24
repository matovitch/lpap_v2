"""Versioned experiment arms (EXPERIMENTS.md). Promoting an arm to a later rung = raising its ``stop_at`` here.

Each arm is one fresh AE run (own HF run folder) from stage 1, 3 and 4 teachers. Within a study every arm shares
the LR schedule (``run.steps``), so rung checkpoints are comparable at equal steps and resume without a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Arm:
    run: str
    overrides: dict[str, Any] = field(default_factory=dict)


SEEDS = (2987, 2988, 2989)


def _noise_arm(study: str, sigma: float, seed: int, stop_at: int, decay: int = 2500) -> Arm:
    tag = "control" if sigma == 0 else f"sig{sigma:g}"
    overrides = {
        "seed": seed,
        "run.steps": 15000,
        "run.stop_at": stop_at,
        "explore.noise_sigma": sigma,
        "explore.noise_decay_steps": decay if sigma > 0 else 0,
    }
    return Arm(f"{study}-{tag}-s{seed}", overrides)


# S1: does annealed relative noise (4 encoder steps) move the AE to a better / different optimum?
# Rung 1 = 5k steps; seed-major order so a partial queue still covers every arm once.
S1 = [_noise_arm("s1", sigma, seed, stop_at=5000) for seed in SEEDS[:2] for sigma in (0.0, 0.3, 1.0, 0.1)]

STUDIES: dict[str, list[Arm]] = {"s1": S1}
