"""Pair geometry: lane table, collision sets and source indices (FOUNDATION §1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int


@dataclass(frozen=True)
class PairSpec:
    """One LPAP pair. ``permutation_seed = None`` means identity permutation."""

    name: str
    bucket_count: int
    k_max: int
    permutation_seed: int | None = None


class PairGeometry:
    """Lane table ``G`` of shape ``(C, P)`` built with a row-major fold.

    ``G[b, p]`` is the energy index at lane ``b``, probe ``p``. It is the single
    source of truth for folding, collision sets and scattering.
    """

    def __init__(self, spec: PairSpec, value_count: int) -> None:
        if value_count <= 0 or spec.bucket_count <= 0:
            raise ValueError("value_count and bucket_count must be positive")
        if value_count % spec.bucket_count:
            raise ValueError("value_count must be divisible by bucket_count")
        if not 1 <= spec.k_max <= spec.bucket_count:
            raise ValueError("k_max must lie in [1, bucket_count]")
        probe_count = value_count // spec.bucket_count
        if probe_count == 1 and spec.k_max != 1:
            raise ValueError("a lossless pair (P = 1) must use k_max = 1")
        self.spec = spec
        self.value_count = value_count
        self.bucket_count = spec.bucket_count
        self.probe_count = probe_count
        self.k_max = spec.k_max
        self.lane_table = lane_table(make_permutation(value_count, spec.permutation_seed), spec.bucket_count)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def is_lossless(self) -> bool:
        return self.probe_count == 1

    @property
    def class_count(self) -> int:
        return self.k_max * self.probe_count

    def to(self, device: torch.device | str) -> PairGeometry:
        self.lane_table = self.lane_table.to(device)
        return self

    def fold(self, values: Float[torch.Tensor, "*batch n"]) -> Float[torch.Tensor, "*batch c p"]:
        return values[..., self.lane_table]

    def candidate_sources(self) -> Int[torch.Tensor, "c k p"]:
        """``s(b, d, p) = G[(b - d) mod C, p]`` for every bucket, dib and probe."""
        return self.lane_table[self.source_lanes()]

    def source_lanes(self) -> Int[torch.Tensor, "c k"]:
        buckets = torch.arange(self.bucket_count, device=self.lane_table.device)
        dibs = torch.arange(self.k_max, device=self.lane_table.device)
        return (buckets[:, None] - dibs[None, :]) % self.bucket_count

    def source_indices(
        self,
        dibs: Int[torch.Tensor, "*batch c"],
        probes: Int[torch.Tensor, "*batch c"],
    ) -> Int[torch.Tensor, "*batch c"]:
        buckets = torch.arange(self.bucket_count, device=dibs.device)
        lanes = (buckets - dibs) % self.bucket_count
        return self.lane_table[lanes, probes]


def make_permutation(value_count: int, seed: int | None) -> Int[torch.Tensor, n]:
    if seed is None:
        return torch.arange(value_count)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(value_count, generator=generator)


def lane_table(permutation: Int[torch.Tensor, n], bucket_count: int) -> Int[torch.Tensor, "c p"]:
    return permutation.reshape(bucket_count, -1).clone()
