"""Gradient routing between energy, surrogate, decoder and decoded energy (FOUNDATION §3).

Every forward value here equals the exact LPAP code. Only the backward
differs from plain autograd, through straight-through terms of the form
``hard + (soft - soft.detach())``, whose forward contribution is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from lpap.geometry import PairGeometry
from lpap.lpap import ExactCode


@dataclass(frozen=True)
class DecoderInputs:
    amplitudes: Float[torch.Tensor, "batch c"]
    dib_weights: Float[torch.Tensor, "batch c k"]
    dibs: Int[torch.Tensor, "batch c"]

    def detach(self) -> DecoderInputs:
        return DecoderInputs(self.amplitudes.detach(), self.dib_weights.detach(), self.dibs)


def visible_candidates(tokens: Float[torch.Tensor, "batch c p"], k_max: int) -> Float[torch.Tensor, "batch c classes"]:
    """``V[b, d·P + p] = tokens[(b - d) mod C, p]``."""
    rolled = torch.stack([torch.roll(tokens, shifts=dib, dims=-2) for dib in range(k_max)], dim=-2)
    return rolled.flatten(-2)


def frozen_call(module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
    """``module(*args)`` with inputs live and params detached: gradients pass through without updating it."""
    params = {name: param.detach() for name, param in module.named_parameters()}
    buffers = dict(module.named_buffers())
    return functional_call(module, (params, buffers), args)


def surrogate_through(surrogate: nn.Module, tokens: Float[torch.Tensor, "batch c p"]) -> torch.Tensor:
    """Frontend call (§3.4): tokens live, surrogate params detached."""
    return frozen_call(surrogate, tokens)


def frontend(
    tokens: Float[torch.Tensor, "batch c p"],
    code: ExactCode,
    geometry: PairGeometry,
    logits: Float[torch.Tensor, "batch c classes"] | None = None,
) -> DecoderInputs:
    """Exact code as decoder inputs (§3.2).

    Forward: ``a = x[i*]``, ``w = one_hot(d*)``. ``a`` always has its direct
    gradient at the winning coordinate. With ``logits``, ``a`` and ``w`` also
    carry the categorical straight-through gradient into the logits.
    """
    choice = F.one_hot(code.local_classes, geometry.class_count).to(tokens.dtype)
    if logits is not None:
        soft = torch.softmax(logits, dim=-1).to(tokens.dtype)
        choice = choice + (soft - soft.detach())
    amplitudes = (choice * visible_candidates(tokens, geometry.k_max)).sum(-1)
    dib_weights = choice.unflatten(-1, (geometry.k_max, geometry.probe_count)).sum(-1)
    return DecoderInputs(amplitudes, dib_weights, code.dibs)


def decoded_energy(
    inputs: DecoderInputs,
    probe_logits: Float[torch.Tensor, "batch c p"],
    geometry: PairGeometry,
) -> Float[torch.Tensor, "batch n"]:
    """One spike per bucket at the decoder's probe, split probe ST (§3.3).

    ``a`` gets its true gradient at the chosen site; ``ℓ`` and ``w`` get the
    gradient of a soft scatter built on ``a.detach()``.
    """
    amplitudes = inputs.amplitudes
    batch = amplitudes.shape[0]
    zeros = amplitudes.new_zeros(batch, geometry.value_count)
    chosen = geometry.source_indices(inputs.dibs, probe_logits.argmax(-1))
    hard = zeros.scatter_add(1, chosen, amplitudes)

    probe_probs = torch.softmax(probe_logits, dim=-1).to(amplitudes.dtype)
    mass = (
        inputs.dib_weights.unsqueeze(-1).to(amplitudes.dtype)
        * probe_probs.unsqueeze(-2)
        * amplitudes.detach()[:, :, None, None]
    )
    sites = geometry.candidate_sources().flatten().expand(batch, -1)
    soft = zeros.scatter_add(1, sites, mass.flatten(1))
    return hard + (soft - soft.detach())
