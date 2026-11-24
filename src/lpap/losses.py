"""Losses and their accuracies (FOUNDATION §4)."""

from __future__ import annotations

import math

import torch
from jaxtyping import Float, Int
from torch import nn
from torch.nn import functional as F

from lpap.lpap import ExactCode

WEIGHT_EPS = 1e-12


def weighted_ce(
    logits: Float[torch.Tensor, "batch c classes"],
    labels: Int[torch.Tensor, "batch c"],
    weights: Float[torch.Tensor, "batch c"],
) -> torch.Tensor:
    """``Σ w·CE / max(Σ w, ε)`` over the whole batch, weights detached, CE in fp32."""
    weights = weights.detach().float()
    ce = F.cross_entropy(logits.float().flatten(0, -2), labels.flatten(), reduction="none")
    return (weights.flatten() * ce).sum() / weights.sum().clamp_min(WEIGHT_EPS)


def accuracies(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain and weighted argmax accuracy, detached."""
    hit = logits.detach().argmax(-1).eq(labels).float()
    weights = weights.detach().float()
    return hit.mean(), (weights * hit).sum() / weights.sum().clamp_min(WEIGHT_EPS)


def surrogate_ce(logits: torch.Tensor, code: ExactCode) -> torch.Tensor:
    return weighted_ce(logits, code.local_classes, code.amplitudes.abs())


def probe_ce(probe_logits: torch.Tensor, code: ExactCode) -> torch.Tensor:
    return weighted_ce(probe_logits, code.probes, code.amplitudes.abs())


def image_l2(reconstruction: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return (reconstruction.float() - image.float()).square().mean()


def signed_mass(energy: Float[torch.Tensor, "batch n"], tau: float) -> torch.Tensor:
    positive = F.relu(energy).mean()
    negative = F.relu(-energy).mean()
    gap = ((positive - negative) / tau).square()
    floor = (F.relu(tau - positive) / tau).square() + (F.relu(tau - negative) / tau).square()
    return gap + floor


def usage_kl(energy: Float[torch.Tensor, "batch n"]) -> torch.Tensor:
    """``KL(p ‖ uniform)`` of the per-coordinate power ``p_i ∝ mean_b x_bi²``: 0 when every coordinate is used equally."""
    power = energy.float().square().mean(0)
    share = power / power.sum().clamp_min(WEIGHT_EPS)
    return (share * (share * share.numel()).clamp_min(WEIGHT_EPS).log()).sum()


def normalized_entropy(counts: Float[torch.Tensor, " n"]) -> torch.Tensor:
    """Entropy of ``counts / Σ counts`` over ``log n``: 1 = uniform, 0 = a single site."""
    share = counts.float() / counts.float().sum().clamp_min(WEIGHT_EPS)
    return -(share * share.clamp_min(WEIGHT_EPS).log()).sum() / math.log(share.numel())


def sample_bridge_time(
    batch: int,
    *,
    eps: float = 1e-4,
    concentration: float = 0.1,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> Float[torch.Tensor, batch]:
    """Half chosen with probability ½, local progress ``u ~ Beta(c, c)`` in ``(ε, 1-ε)``."""
    gamma_a = torch._standard_gamma(torch.full((batch,), concentration), generator=generator)
    gamma_b = torch._standard_gamma(torch.full((batch,), concentration), generator=generator)
    progress = (gamma_a / (gamma_a + gamma_b)).nan_to_num(0.5).clamp(eps, 1.0 - eps)
    left = torch.rand(batch, generator=generator) < 0.5
    return torch.where(left, progress - 1.0, progress).to(device)


def flow_matching(
    field: nn.Module,
    image: Float[torch.Tensor, "batch 1 n"],
    prior: Float[torch.Tensor, "batch 1 n"],
    time: Float[torch.Tensor, batch],
) -> torch.Tensor:
    """Bidirectional bridge: ``t=-1`` image, ``t=0`` prior, ``t=1`` image."""
    left = (time <= 0)[:, None, None]
    alpha = torch.where(left, time[:, None, None] + 1.0, 1.0 - time[:, None, None])
    state = (1.0 - alpha) * image + alpha * prior
    target = torch.where(left, prior - image, image - prior)
    return (field(state, time) - target).float().square().mean()


def round_trip_l2(
    field: nn.Module, image: Float[torch.Tensor, "batch 1 n"], steps: int, *, bf16: bool = False
) -> torch.Tensor:
    """``‖render(encode(I)) − I‖²`` with ``steps`` Euler steps each way (differentiable; fp32 state)."""
    from lpap.flow import integrate

    energy = integrate(field, image, t0=-1.0, t1=0.0, steps=steps, bf16=bf16)
    return (integrate(field, energy, t0=0.0, t1=1.0, steps=steps, bf16=bf16) - image).square().mean()
