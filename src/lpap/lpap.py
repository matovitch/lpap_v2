"""Exact LPAP operator (FOUNDATION §1, §2.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int

from lpap.geometry import PairGeometry


@dataclass(frozen=True)
class LPAPTokenResult:
    """Exact LPAP on folded ``(..., C, P)`` tokens."""

    amplitudes: Float[torch.Tensor, "*batch c"]
    dibs: Int[torch.Tensor, "*batch c"]
    probes: Int[torch.Tensor, "*batch c"]
    remaining: Float[torch.Tensor, "*batch c p"]


@dataclass(frozen=True)
class ExactCode:
    """Exact LPAP code of an energy for one pair. All fields carry no gradient."""

    amplitudes: Float[torch.Tensor, "batch c"]
    dibs: Int[torch.Tensor, "batch c"]
    probes: Int[torch.Tensor, "batch c"]
    sources: Int[torch.Tensor, "batch c"]
    local_classes: Int[torch.Tensor, "batch c"]


def validate_tokens(tokens: torch.Tensor, k_max: int) -> None:
    if tokens.ndim < 2:
        raise ValueError("tokens must have at least two dimensions")
    bucket_count, probe_count = tokens.shape[-2:]
    if bucket_count <= 0 or probe_count <= 0:
        raise ValueError("bucket and probe counts must be positive")
    if not 1 <= k_max <= bucket_count:
        raise ValueError("k_max must lie in [1, bucket_count]")
    if not tokens.dtype.is_floating_point:
        raise TypeError("tokens must be floating point")


def lpap_tokens_torch(tokens: Float[torch.Tensor, "*batch c p"], *, k_max: int) -> LPAPTokenResult:
    """Reference operator. Fixed-shape roll/gather/where updates, no host syncs.

    Per roll ``k``, bucket ``i`` reads lane ``(i - k) mod C`` and picks the
    largest ``|value|`` among valid slots (lowest probe on ties). It takes the
    candidate when empty or when ``|cand| >= |bucket|``; a displaced value
    swaps back into the vacated slot with a dib offset.
    """
    validate_tokens(tokens, k_max)
    batch_shape = tokens.shape[:-2]
    bucket_count, probe_count = tokens.shape[-2:]
    batch_count = math.prod(batch_shape) if batch_shape else 1
    device, dtype = tokens.device, tokens.dtype
    shape = (batch_count, bucket_count, probe_count)

    work = tokens.reshape(shape).clone()
    work_valid = torch.ones(shape, device=device, dtype=torch.bool)
    probe_arange = torch.arange(probe_count, device=device).view(1, 1, probe_count)
    work_probes = probe_arange.expand(shape).clone()
    dib_offsets = torch.zeros(shape, device=device, dtype=torch.int64)

    amplitudes = torch.zeros((batch_count, bucket_count), device=device, dtype=dtype)
    bucket_valid = torch.zeros((batch_count, bucket_count), device=device, dtype=torch.bool)
    dibs = torch.zeros((batch_count, bucket_count), device=device, dtype=torch.int64)
    probes = torch.zeros_like(dibs)
    zero_value = torch.zeros_like(amplitudes)
    zero_index = torch.zeros_like(dibs)
    neg_one = torch.tensor(-1.0, device=device, dtype=dtype)

    for roll in range(k_max):
        lane_values = torch.roll(work, shifts=roll, dims=1)
        lane_valid = torch.roll(work_valid, shifts=roll, dims=1)
        lane_offsets = torch.roll(dib_offsets, shifts=roll, dims=1)
        lane_probes = torch.roll(work_probes, shifts=roll, dims=1)

        finite_valid = lane_valid & ~torch.isnan(lane_values)
        scores = torch.where(finite_valid, lane_values.abs(), neg_one)
        position = scores.argmax(dim=-1, keepdim=True)

        candidates = lane_values.gather(-1, position).squeeze(-1)
        candidate_valid = finite_valid.gather(-1, position).squeeze(-1)
        candidate_dibs = lane_offsets.gather(-1, position).squeeze(-1) + roll
        candidate_probes = lane_probes.gather(-1, position).squeeze(-1)

        take = candidate_valid & (~bucket_valid | (candidates.abs() >= amplitudes.abs()))
        slot = (probe_arange == position) & take.unsqueeze(-1)

        swap_values = torch.where(bucket_valid, amplitudes, zero_value).unsqueeze(-1)
        swap_offsets = torch.where(bucket_valid, dibs - roll, zero_index).unsqueeze(-1)
        swap_probes = torch.where(bucket_valid, probes, zero_index).unsqueeze(-1)

        work = torch.roll(torch.where(slot, swap_values, lane_values), -roll, dims=1)
        work_valid = torch.roll(torch.where(slot, bucket_valid.unsqueeze(-1), lane_valid), -roll, dims=1)
        dib_offsets = torch.roll(torch.where(slot, swap_offsets, lane_offsets), -roll, dims=1)
        work_probes = torch.roll(torch.where(slot, swap_probes, lane_probes), -roll, dims=1)

        amplitudes = torch.where(take, candidates, amplitudes)
        bucket_valid = bucket_valid | take
        dibs = torch.where(take, candidate_dibs, dibs)
        probes = torch.where(take, candidate_probes, probes)

    return LPAPTokenResult(
        amplitudes=amplitudes.reshape(*batch_shape, bucket_count),
        dibs=dibs.reshape(*batch_shape, bucket_count),
        probes=probes.reshape(*batch_shape, bucket_count),
        remaining=torch.where(work_valid, work, zero_value.unsqueeze(-1)).reshape(tokens.shape),
    )


def lpap_tokens(tokens: Float[torch.Tensor, "*batch c p"], *, k_max: int, use_triton: bool = True) -> LPAPTokenResult:
    """Triton on CUDA when available and requested, torch reference otherwise."""
    if use_triton and tokens.is_cuda:
        from lpap.lpap_triton import lpap_tokens_triton

        return lpap_tokens_triton(tokens, k_max=k_max)
    return lpap_tokens_torch(tokens, k_max=k_max)


@torch.no_grad()
def exact_code(
    values: Float[torch.Tensor, "batch n"],
    geometry: PairGeometry,
    *,
    use_triton: bool = True,
) -> ExactCode:
    """Exact LPAP code of detached energies for one pair."""
    tokens = geometry.fold(values.detach())
    result = lpap_tokens(tokens, k_max=geometry.k_max, use_triton=use_triton)
    sources = geometry.source_indices(result.dibs, result.probes)
    return ExactCode(
        amplitudes=result.amplitudes,
        dibs=result.dibs,
        probes=result.probes,
        sources=sources,
        local_classes=result.dibs * geometry.probe_count + result.probes,
    )
