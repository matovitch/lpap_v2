"""Triton CUDA kernel for exact LPAP, bit-identical to ``lpap_tokens_torch``."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from jaxtyping import Float

from lpap.lpap import LPAPTokenResult, validate_tokens


@triton.jit
def _lpap_tokens_kernel(
    values_ptr,
    work_ptr,
    work_valid_ptr,
    offset_ptr,
    work_probe_ptr,
    amplitudes_ptr,
    bucket_valid_ptr,
    dibs_ptr,
    probes_ptr,
    value_count,
    bucket_count,
    probe_count,
    k_max,
    block_probe_count: tl.constexpr,
):
    batch_index = tl.program_id(0).to(tl.int64)
    probe_count_i64 = tl.cast(probe_count, tl.int64)
    value_base = batch_index * tl.cast(value_count, tl.int64)
    bucket_base = batch_index * tl.cast(bucket_count, tl.int64)

    value_dtype = values_ptr.dtype.element_ty
    zero_val = tl.full([], 0.0, dtype=value_dtype)
    neg_one_vec = tl.full([block_probe_count], -1.0, dtype=value_dtype)
    zero_i64 = tl.full([], 0, dtype=tl.int64)
    zeros_i64_vec = tl.full([block_probe_count], 0, dtype=tl.int64)
    zero_i8 = tl.full([], 0, dtype=tl.int8)
    one_i8 = tl.full([], 1, dtype=tl.int8)
    ones_i8_vec = tl.full([block_probe_count], 1, dtype=tl.int8)

    probe_offsets = tl.arange(0, block_probe_count).to(tl.int64)
    probe_mask = probe_offsets < probe_count_i64
    sentinel = tl.full([block_probe_count], block_probe_count, dtype=tl.int64)

    for bucket_index in range(0, bucket_count):
        bucket_i64 = bucket_index.to(tl.int64)
        offsets = value_base + bucket_i64 * probe_count_i64 + probe_offsets
        lane = tl.load(values_ptr + offsets, mask=probe_mask, other=zero_val)
        tl.store(work_ptr + offsets, lane, mask=probe_mask)
        tl.store(work_valid_ptr + offsets, ones_i8_vec, mask=probe_mask)
        tl.store(offset_ptr + offsets, zeros_i64_vec, mask=probe_mask)
        tl.store(work_probe_ptr + offsets, probe_offsets, mask=probe_mask)
        bucket_offset = bucket_base + bucket_i64
        tl.store(amplitudes_ptr + bucket_offset, zero_val)
        tl.store(bucket_valid_ptr + bucket_offset, zero_i8)
        tl.store(dibs_ptr + bucket_offset, zero_i64)
        tl.store(probes_ptr + bucket_offset, zero_i64)

    for roll in range(0, k_max):
        roll_i64 = roll.to(tl.int64)
        for bucket_index in range(0, bucket_count):
            bucket_i64 = bucket_index.to(tl.int64)
            source_lane = (bucket_index - roll + bucket_count) % bucket_count
            source_base = value_base + source_lane.to(tl.int64) * probe_count_i64
            source_offsets = source_base + probe_offsets

            lane_values = tl.load(work_ptr + source_offsets, mask=probe_mask, other=zero_val)
            lane_valid = tl.load(work_valid_ptr + source_offsets, mask=probe_mask, other=zero_i8)
            is_valid = probe_mask & (lane_valid != zero_i8) & (lane_values == lane_values)
            scores = tl.where(is_valid, tl.abs(lane_values), neg_one_vec)
            max_score = tl.max(scores, axis=0)
            has_valid = max_score >= zero_val
            first = tl.min(tl.where((scores == max_score) & is_valid, probe_offsets, sentinel), axis=0)
            candidate_index = tl.where(has_valid, first, zero_i64)
            candidate_offset = source_base + candidate_index

            candidate = tl.load(work_ptr + candidate_offset, mask=has_valid, other=zero_val)
            candidate_offset_dib = tl.load(offset_ptr + candidate_offset, mask=has_valid, other=zero_i64)
            candidate_probe = tl.load(work_probe_ptr + candidate_offset, mask=has_valid, other=zero_i64)

            bucket_offset = bucket_base + bucket_i64
            old_amplitude = tl.load(amplitudes_ptr + bucket_offset)
            old_valid = tl.load(bucket_valid_ptr + bucket_offset)
            old_dib = tl.load(dibs_ptr + bucket_offset)
            old_probe = tl.load(probes_ptr + bucket_offset)
            old_is_valid = old_valid != zero_i8
            take = has_valid & ((~old_is_valid) | (tl.abs(candidate) >= tl.abs(old_amplitude)))

            tl.store(work_ptr + candidate_offset, tl.where(old_is_valid, old_amplitude, zero_val), mask=take)
            tl.store(work_valid_ptr + candidate_offset, old_valid, mask=take)
            tl.store(offset_ptr + candidate_offset, tl.where(old_is_valid, old_dib - roll_i64, zero_i64), mask=take)
            tl.store(work_probe_ptr + candidate_offset, tl.where(old_is_valid, old_probe, zero_i64), mask=take)
            tl.store(amplitudes_ptr + bucket_offset, candidate, mask=take)
            tl.store(bucket_valid_ptr + bucket_offset, one_i8, mask=take)
            tl.store(dibs_ptr + bucket_offset, candidate_offset_dib + roll_i64, mask=take)
            tl.store(probes_ptr + bucket_offset, candidate_probe, mask=take)


def lpap_tokens_triton(tokens: Float[torch.Tensor, "*batch c p"], *, k_max: int) -> LPAPTokenResult:
    if tokens.requires_grad:
        raise RuntimeError("lpap_tokens_triton has no autograd; detach the tokens")
    if not tokens.is_cuda:
        raise ValueError("lpap_tokens_triton needs CUDA tokens")
    validate_tokens(tokens, k_max)
    batch_shape = tokens.shape[:-2]
    bucket_count, probe_count = tokens.shape[-2:]
    batch_count = math.prod(batch_shape) if batch_shape else 1
    value_count = bucket_count * probe_count
    device = tokens.device

    flat = tokens.reshape(batch_count, value_count).contiguous()
    work = torch.empty_like(flat)
    work_valid = torch.empty((batch_count, value_count), device=device, dtype=torch.int8)
    offsets = torch.empty((batch_count, value_count), device=device, dtype=torch.int64)
    work_probes = torch.empty_like(offsets)
    amplitudes = torch.empty((batch_count, bucket_count), device=device, dtype=tokens.dtype)
    bucket_valid = torch.empty((batch_count, bucket_count), device=device, dtype=torch.int8)
    dibs = torch.empty((batch_count, bucket_count), device=device, dtype=torch.int64)
    probes = torch.empty_like(dibs)

    if batch_count > 0:
        _lpap_tokens_kernel[(batch_count,)](
            flat, work, work_valid, offsets, work_probes,
            amplitudes, bucket_valid, dibs, probes,
            value_count, bucket_count, probe_count, k_max,
            triton.next_power_of_2(probe_count),
        )  # fmt: skip
    remaining = torch.where(work_valid.bool(), work, torch.zeros_like(work))
    return LPAPTokenResult(
        amplitudes=amplitudes.reshape(*batch_shape, bucket_count),
        dibs=dibs.reshape(*batch_shape, bucket_count),
        probes=probes.reshape(*batch_shape, bucket_count),
        remaining=remaining.reshape(tokens.shape),
    )
