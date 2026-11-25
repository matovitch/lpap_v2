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


def _noise_arm(study: str, sigma: float, seed: int, stop_at: int, decay: int = 2500, steps: int = 15000) -> Arm:
    tag = "control" if sigma == 0 else f"sig{sigma:g}"
    overrides = {
        "seed": seed,
        "run.steps": steps,
        "run.stop_at": stop_at,
        "explore.noise_sigma": sigma,
        "explore.noise_decay_steps": decay if sigma > 0 else 0,
    }
    return Arm(f"{study}-{tag}-s{seed}", overrides)


# S1: does annealed relative noise (4 encoder steps) move the AE to a better / different optimum?
# Rung 1 = 5k steps; seed-major order so a partial queue still covers every arm once.
S1 = [_noise_arm("s1", sigma, seed, stop_at=5000) for seed in SEEDS[:2] for sigma in (0.0, 0.3, 1.0, 0.1)]

# Rung 2 = 15k for every arm (none worse than control beyond the seed spread at 5k). One seed per molab
# session (≈ 4 h each): resumes each run from its 5k ``last.pt`` on HF.
S1_R2 = {seed: [_noise_arm("s1", sigma, seed, stop_at=15000) for sigma in (0.0, 0.1, 0.3, 1.0)] for seed in SEEDS[:2]}


def _anytime_arm(
    study: str,
    tag: str,
    seed: int,
    stop_at: int,
    *,
    noise_mode: str = "linear",
    sigma: float = 0.0,
    half_life: float = 0.0,
) -> Arm:
    overrides: dict[str, Any] = {
        "optimizer": "schedulefree",
        "seed": seed,
        "run.steps": stop_at,
        "run.stop_at": stop_at,
        "explore.noise_sigma": sigma,
        "explore.noise_decay_steps": 0,
        "explore.noise_mode": noise_mode,
        "explore.noise_half_life": half_life,
    }
    return Arm(f"{study}-{tag}-s{seed}", overrides)


# S2: anytime training — Schedule-Free AdamW + gradient-driven relative noise (σ_max = 1.0, shared h = 500).
# Nothing in the optimizer or noise controller depends on ``run.steps`` (which is only the truncation budget).
# One seed per molab session (3 × 15k ≈ 4.5 h).
S2 = {
    seed: [
        _anytime_arm("s2", "sf-control", seed, stop_at=15000),
        _anytime_arm("s2", "sf-gnorm", seed, stop_at=15000, noise_mode="grad_norm", sigma=1.0, half_life=500.0),
        _anytime_arm("s2", "sf-gsnr", seed, stop_at=15000, noise_mode="grad_snr", sigma=1.0, half_life=500.0),
    ]
    for seed in SEEDS[:2]
}

STUDIES: dict[str, list[Arm]] = {
    "s1": S1,
    "s1-r2-s2988": S1_R2[2988],
    "s1-r2-s2987": [arm for arm in S1_R2[2987] if arm.overrides["explore.noise_sigma"] in (0.0, 1.0)],  # confirmation
    "s2-s2988": S2[2988],
    "s2-s2987": S2[2987],
}
