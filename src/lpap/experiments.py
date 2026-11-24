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


def _noise_arm(
    study: str, sigma: float, seed: int, stop_at: int, decay: int = 2500, steps: int = 15000
) -> Arm:
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
S1_R2 = {
    seed: [_noise_arm("s1", sigma, seed, stop_at=15000) for sigma in (0.0, 0.1, 0.3, 1.0)] for seed in SEEDS[:2]
}

# S2: long noise, a dose ladder (dose ∝ σ0² · decay; ×4 per level, σ0 0.5 = S1's σ0 1 dose). 20k schedule,
# 10k linear decay, judged at 20k. Batch A = the most informative arms (≈ 6 h), batch B = the same-dose arm.
S2_A = [_noise_arm("s2", sigma, 2988, stop_at=20000, decay=10000, steps=20000) for sigma in (0.0, 1.0, 2.0)]
S2_B = [_noise_arm("s2", 0.5, 2988, stop_at=20000, decay=10000, steps=20000)]

STUDIES: dict[str, list[Arm]] = {
    "s1": S1,
    "s1-r2-s2988": S1_R2[2988],
    "s1-r2-s2987": S1_R2[2987],
    "s2-a": S2_A,
    "s2-b": S2_B,
}
