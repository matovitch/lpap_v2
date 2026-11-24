"""Stage 5: the autoencoder over all pairs (FOUNDATION §3, §4.4–4.6, §5).

Forward = inference: encode 4 steps, exact LPAP per pair, decoder, scatter,
render 4 steps. Gradients follow §3: categorical ST through the frozen-param
surrogate call, split probe ST in the scatter, surrogate and decoder CEs on
detached inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lpap.codec import decoded_energy, frontend, frozen_call, surrogate_through
from lpap.data import ImageSet
from lpap.flow import integrate
from lpap.gallery import energy_tiles, grid, image_tiles, write_png
from lpap.geometry import PairGeometry
from lpap.losses import (
    accuracies,
    centered_cosine,
    image_l2,
    normalized_entropy,
    probe_ce,
    signed_mass,
    surrogate_ce,
    usage_kl,
)
from lpap.lpap import exact_code
from lpap.runtime import checkpoint
from lpap.runtime.config import ModelConfig, to_dict
from lpap.runtime.hub import ArtifactStore
from lpap.runtime.run import BEST, ProgressFn, RunConfig, RunResult, run
from lpap.stages import decoder as decoder_stage
from lpap.stages import surrogate as surrogate_stage
from lpap.stages.flow import learning_rate, load_field, top_share

RUN = "ae"
GROUPS = ("encoder", "renderer", "surrogates", "decoders")
FOOTPRINT_SITES = 16  # energy coordinates probed per image for the gathering footprint


def gather_footprint(
    encode: Any, image: torch.Tensor, sites: int = FOOTPRINT_SITES
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which inputs each energy coordinate gathers from: rows ``J_i = ∂x_i/∂input`` of the encoder Jacobian.

    ``area`` = participation ratio ``(Σ|J_i|)² / Σ J_i²`` (≈ number of input samples used; 1 = a single one);
    ``span`` = ``J_i²``-weighted rms distance to ``i`` along the 1-D signal. Means over images and probed sites
    (``sites`` evenly spaced coordinates). Signal-agnostic: only the flattened order is used.
    """
    batch, _, n = image.shape
    coords = torch.arange(sites, device=image.device) * (n // sites) + n // (2 * sites)
    inputs = image.repeat_interleave(sites, 0).detach().requires_grad_(True)
    with torch.enable_grad():
        energy = encode(inputs)
        picked = energy.gather(1, coords.repeat(batch)[:, None]).sum()
        (jacobian,) = torch.autograd.grad(picked, inputs)
    rows = jacobian.squeeze(1).float().abs()
    power = rows.square()
    area = rows.sum(-1).square() / power.sum(-1).clamp_min(1e-30)
    offset = torch.arange(n, device=image.device)[None, :] - coords.repeat(batch)[:, None]
    span = ((power * offset.square()).sum(-1) / power.sum(-1).clamp_min(1e-30)).sqrt()
    return area.mean(), span.mean()


@dataclass(frozen=True)
class SideTerms:
    """Candidate side terms (§4.7), in priority order. Weight 0 = monitor only; gradient reaches the encoder only."""

    residual_share: float
    probe_ce_live: float
    usage_kl: float


@dataclass(frozen=True)
class Explore:
    """Exploration knobs (EXPERIMENTS.md). Defaults (0, 0, None) = the plain AE.

    Annealed relative noise: training only, ``x̃ = x + σ(step)·rms(x).detach()·ε`` feeds LPAP and the
    renderer for every pair; ``σ`` decays linearly from ``noise_sigma`` to 0 over ``noise_decay_steps``.
    """

    noise_sigma: float
    noise_decay_steps: int
    encoder_steps: int | None  # None: ``integration_steps``; the renderer always uses ``integration_steps``

    def sigma(self, step: int) -> float:
        if self.noise_sigma <= 0 or self.noise_decay_steps <= 0:
            return 0.0
        return self.noise_sigma * max(0.0, 1.0 - step / self.noise_decay_steps)


@dataclass(frozen=True)
class AEStageConfig:
    seed: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    min_lr_ratio: float
    max_grad_norm: float
    compile: bool
    bf16: bool
    integration_steps: int
    surrogate_weight: float
    probe_weight: float
    signed_mass_weight: float
    signed_mass_tau: float
    surrogate_gradient: bool
    side: SideTerms
    explore: Explore
    validation_batch_size: int
    validation_batch_count: int
    gallery_count: int
    run: RunConfig


def check_pair(store: ArtifactStore, run_name: str, geometry: PairGeometry) -> None:
    """The teacher checkpoint must have been trained on the same pair spec (hence the same ``G``)."""
    path = store.fetch(run_name, BEST)
    if path is None:
        raise FileNotFoundError(f"no best checkpoint for {run_name}")
    config = checkpoint.load(path)["config"]
    specs = {pair["name"]: pair for pair in config["model"]["pairs"]}
    if specs.get(geometry.name) != to_dict(geometry.spec):
        raise ValueError(f"{run_name} was trained on {specs.get(geometry.name)}, not {to_dict(geometry.spec)}")


class AETask:
    best_metric = "image_l2"

    def __init__(
        self,
        model: ModelConfig,
        config: AEStageConfig,
        geometries: dict[str, PairGeometry],
        images: ImageSet,
        store: ArtifactStore,
    ) -> None:
        self.config = config
        self.geometries = geometries
        self.lossy = [name for name, geometry in geometries.items() if not geometry.is_lossless]
        self.images = images
        self.device = images.device
        torch.manual_seed(config.seed)
        self.encoder = load_field(model, store, self.device)
        self.renderer = load_field(model, store, self.device)
        self.surrogates = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for name in self.lossy:
            geometry = geometries[name]
            check_pair(store, surrogate_stage.run_name(name), geometry)
            check_pair(store, decoder_stage.run_name(name), geometry)
            self.surrogates[name] = surrogate_stage.load_surrogate(model, geometry, store, self.device)
            self.decoders[name] = decoder_stage.load_decoder(model, geometry, store, self.device)
        self.modules = nn.ModuleDict(
            {
                "encoder": self.encoder,
                "renderer": self.renderer,
                "surrogates": self.surrogates,
                "decoders": self.decoders,
            }
        )
        compiled = torch.compile if config.compile else (lambda module: module)
        self.run_encoder = compiled(self.encoder)
        self.run_renderer = compiled(self.renderer)
        self.run_surrogates = {name: compiled(module) for name, module in self.surrogates.items()}
        self.run_decoders = {name: compiled(module) for name, module in self.decoders.items()}
        self.optimizer = torch.optim.AdamW(self.modules.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step = 0
        self.validation_images = images.validation_batches(config.validation_batch_size, config.validation_batch_count)
        self.stage1_energies, self.stage1_footprint = self._stage1_reference(model, store)

    def _footprint_images(self) -> torch.Tensor:
        return self.validation_images[0][:4]

    def _stage1_reference(
        self, model: ModelConfig, store: ArtifactStore
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        """Frozen stage-1 flow on the validation images: energies for ``stage1_corr`` and its gathering footprint."""
        field = load_field(model, store, self.device).eval()
        steps = self.config.integration_steps

        def encode(image: torch.Tensor) -> torch.Tensor:
            return integrate(field, image, t0=-1.0, t1=0.0, steps=steps, bf16=False).squeeze(1)

        with torch.no_grad():
            energies = [encode(image).float() for image in self.validation_images]
        area, span = gather_footprint(encode, self._footprint_images())
        del field
        return energies, {"gather_area_stage1": float(area), "gather_span_stage1": float(span)}

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.config.bf16)

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        steps = self.config.explore.encoder_steps or self.config.integration_steps
        return integrate(self.run_encoder, image, t0=-1.0, t1=0.0, steps=steps, bf16=self.config.bf16).squeeze(1)

    def _render(self, decoded: list[torch.Tensor]) -> list[torch.Tensor]:
        """One renderer call for all pairs (batch ``pairs · B``)."""
        steps = self.config.integration_steps
        stacked = torch.cat(decoded).unsqueeze(1)
        return list(
            integrate(self.run_renderer, stacked, t0=0.0, t1=1.0, steps=steps, bf16=self.config.bf16).chunk(
                len(decoded)
            )
        )

    def _lossy(
        self, name: str, energy: torch.Tensor, *, train: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Decoded energy of one lossy pair, its CE terms and diagnostics, and its live side terms (§4.7)."""
        geometry = self.geometries[name]
        tokens = geometry.fold(energy)
        code = exact_code(energy, geometry)
        weights = code.amplitudes.abs()
        with self._autocast():
            surrogate_logits = self.run_surrogates[name](tokens.detach())
            routed = None
            if train and self.config.surrogate_gradient:
                routed = surrogate_through(self.surrogates[name], tokens)
        inputs = frontend(tokens, code, geometry, None if routed is None else routed.float())
        detached = inputs.detach()
        batch = energy.shape[0]
        with self._autocast():
            # Live inputs for the image loss and detached ones for the probe CE, in one decoder call.
            probe_logits = self.run_decoders[name](
                torch.cat([inputs.amplitudes, detached.amplitudes]),
                torch.cat([inputs.dib_weights, detached.dib_weights]),
            ).float()
        live_logits, ce_logits = probe_logits[:batch], probe_logits[batch:]
        decoded = decoded_energy(inputs, live_logits, geometry)
        surrogate_accuracy, surrogate_weighted = accuracies(surrogate_logits, code.local_classes, weights)
        probe_accuracy, probe_weighted = accuracies(ce_logits, code.probes, weights)
        kept = code.amplitudes.square().sum(-1) / energy.detach().square().sum(-1).clamp_min(1e-12)
        kept_sites = (
            energy.new_zeros(energy.shape)
            .scatter_add_(1, code.sources, code.amplitudes.ne(0).to(energy.dtype))
            .clamp_max(1.0)
        )
        power = energy.square()
        probe_live = probe_ce(live_logits, code)
        if train and self.config.side.probe_ce_live > 0:
            # Enabled: frozen decoder params, so this term trains the encoder only.
            with self._autocast():
                probe_live = probe_ce(
                    frozen_call(self.decoders[name], inputs.amplitudes, inputs.dib_weights).float(), code
                )
        side = {
            # 1 − kept_share, with a live gradient: pushes ‖x‖² from dropped onto kept coordinates.
            "residual_share": ((power * (1.0 - kept_sites)).sum(-1) / power.sum(-1).clamp_min(1e-12)).mean(),
            "probe_ce_live": probe_live,
        }
        terms = {
            "surrogate_ce": surrogate_ce(surrogate_logits, code),
            "probe_ce": probe_ce(ce_logits, code),
            "surrogate_accuracy": surrogate_accuracy,
            "surrogate_weighted_accuracy": surrogate_weighted,
            "probe_accuracy": probe_accuracy,
            "probe_weighted_accuracy": probe_weighted,
            "kept_share": kept.mean(),
            "support_entropy": normalized_entropy(kept_sites.mean(0)),
        }
        return decoded, terms, side

    def _forward(
        self, image: torch.Tensor, *, train: bool, noise: float = 0.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], list, dict[str, torch.Tensor]]:
        """``metrics["loss"]`` is the main objective (§4.6); side terms come back live, mean over lossy pairs.

        ``noise``: relative σ of the exploration noise on the energy seen by LPAP and the renderer.
        """
        config = self.config
        energy = self._encode(image)
        seen = energy
        if noise > 0:
            rms = energy.detach().square().mean(-1, keepdim=True).sqrt()
            seen = energy + noise * rms * torch.randn_like(energy)
        decoded, metrics = [], {}
        pair_sides: list[dict[str, torch.Tensor]] = []
        for name, geometry in self.geometries.items():
            if geometry.is_lossless:
                decoded.append(seen)
                continue
            pair_decoded, terms, pair_side = self._lossy(name, seen, train=train)
            decoded.append(pair_decoded)
            pair_sides.append(pair_side)
            metrics.update({f"{key}_{name}": value for key, value in terms.items()})
        renders = self._render(decoded)
        for name, render in zip(self.geometries, renders, strict=True):
            metrics[f"image_l2_{name}"] = image_l2(render, image)
        metrics["image_l2"] = torch.stack([metrics[f"image_l2_{name}"] for name in self.geometries]).mean()
        metrics["signed_mass"] = signed_mass(energy, config.signed_mass_tau)
        metrics["positive_mass"] = energy.detach().relu().mean()
        metrics["negative_mass"] = (-energy.detach()).relu().mean()
        loss = metrics["image_l2"]
        if config.signed_mass_weight > 0:
            loss = loss + config.signed_mass_weight * metrics["signed_mass"]
        if self.lossy:
            for key, weight in (("surrogate_ce", config.surrogate_weight), ("probe_ce", config.probe_weight)):
                metrics[key] = torch.stack([metrics[f"{key}_{name}"] for name in self.lossy]).mean()
                loss = loss + weight * metrics[key]
        metrics["loss"] = loss
        side = {key: torch.stack([pair[key] for pair in pair_sides]).mean() for key in (pair_sides or [{}])[0]}
        side["usage_kl"] = usage_kl(energy)
        metrics["usage_kl"] = side["usage_kl"].detach()
        return energy, metrics, renders, side

    def _side_loss(self, side: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Enabled side terms, weighted and summed (None when all weights are 0)."""
        enabled = [
            getattr(self.config.side, name) * term for name, term in side.items() if getattr(self.config.side, name) > 0
        ]
        return torch.stack(enabled).sum() if enabled else None

    def train_step(self) -> dict[str, torch.Tensor | float]:
        self.modules.train()
        rate = learning_rate(self.config, self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        image = self.images.sample_train(self.config.batch_size, self.generator)
        noise = self.config.explore.sigma(self.step)
        _, metrics, _, side = self._forward(image, train=True, noise=noise)
        self.optimizer.zero_grad(set_to_none=True)
        # Side terms reach the encoder only by construction (§4.7), so one backward serves both.
        side_loss = self._side_loss(side)
        if side_loss is not None:
            metrics["side_loss"] = side_loss.detach()
            metrics["loss"] = metrics["loss"] + side_loss
        metrics["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.modules.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.step += 1
        return {
            **{name: value.detach() for name, value in metrics.items()},
            "grad_norm": grad_norm,
            "lr": rate,
            "noise_sigma": noise,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Fixed validation images. ``kept_share_<pair>``: share of ``‖x‖²`` the exact code keeps."""
        self.modules.eval()
        totals: dict[str, float] = {}
        for image, reference in zip(self.validation_images, self.stage1_energies, strict=True):
            energy, metrics, _, _ = self._forward(image, train=False)
            metrics["stage1_corr"] = centered_cosine(energy.float(), reference).mean()
            metrics["energy_mean"] = energy.mean()
            metrics["energy_std"] = energy.std()
            metrics["energy_top10"] = top_share(energy, 0.1)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) / len(self.validation_images)
        totals.update(self.gradient_norms(self.validation_images[0]))
        totals.update(self.footprint())
        return totals

    def footprint(self) -> dict[str, float]:
        """Gathering footprint of the live encoder (eager, fp32) next to its stage-1 reference."""
        steps = self.config.explore.encoder_steps or self.config.integration_steps

        def encode(image: torch.Tensor) -> torch.Tensor:
            return integrate(self.encoder, image, t0=-1.0, t1=0.0, steps=steps, bf16=False).squeeze(1)

        area, span = gather_footprint(encode, self._footprint_images())
        return {"gather_area": float(area), "gather_span": float(span), **self.stage1_footprint}

    def gradient_norms(self, image: torch.Tensor) -> dict[str, float]:
        """``grad_<term>_<module>``: ‖∂term/∂θ_module‖ of each unweighted term, as routed (§3.6, §4.7).

        Main terms as in the training graph, side terms through their live path. A missing
        entry means no gradient reaches that module. Runs the eager modules: compiled
        backwards donate their buffers and cannot be replayed (``retain_graph``).
        """
        self.modules.eval()
        compiled = self.run_encoder, self.run_renderer, self.run_surrogates, self.run_decoders
        self.run_encoder, self.run_renderer = self.encoder, self.renderer
        self.run_surrogates, self.run_decoders = dict(self.surrogates.items()), dict(self.decoders.items())
        try:
            return self._gradient_norms(image)
        finally:
            self.run_encoder, self.run_renderer, self.run_surrogates, self.run_decoders = compiled

    def _gradient_norms(self, image: torch.Tensor) -> dict[str, float]:
        with torch.enable_grad():
            _, metrics, _, side = self._forward(image, train=True)
            terms = {
                key: metrics[key] for key in ("image_l2", "signed_mass", "surrogate_ce", "probe_ce") if key in metrics
            }
            terms.update(side)
            params = {group: [p for p in self.modules[group].parameters() if p.requires_grad] for group in GROUPS}
            flat = [p for group in GROUPS for p in params[group]]
            norms: dict[str, float] = {}
            for term_name, term in terms.items():
                grads = iter(torch.autograd.grad(term, flat, retain_graph=True, allow_unused=True))
                for group in GROUPS:
                    found = [g.norm() for g in (next(grads) for _ in params[group]) if g is not None]
                    # Zero, not just None: slices of a shared call give exact-zero gradients.
                    norm = float(torch.linalg.vector_norm(torch.stack(found))) if found else 0.0
                    if norm > 0:
                        norms[f"grad_{term_name}_{group}"] = norm
        return norms

    @torch.no_grad()
    def gallery(self, path: Path) -> Path:
        """Rows: images, energies, then one reconstruction row per pair (model config order)."""
        self.modules.eval()
        image = self.validation_images[0][: self.config.gallery_count]
        energy, _, renders, _ = self._forward(image, train=False)
        return write_png(path, grid([image_tiles(image), energy_tiles(energy), *map(image_tiles, renders)]))

    def state_dict(self) -> dict[str, Any]:
        return {
            "modules": self.modules.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.modules.load_state_dict(state["modules"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.step = state["step"]

    def weights(self) -> dict[str, Any]:
        return {"modules": self.modules.state_dict()}


def train(
    model: ModelConfig,
    config: AEStageConfig,
    geometries: dict[str, PairGeometry],
    images: ImageSet,
    store: ArtifactStore,
    progress: ProgressFn | None = None,
    *,
    name: str = RUN,
) -> RunResult:
    """Train every pair jointly into run ``name``; teachers and flow come from stages 1, 3 and 4."""
    task = AETask(model, config, geometries, images, store)
    resolved = {"model": to_dict(model), "stage": to_dict(config), "pairs": list(geometries)}
    return run(task, run=name, config=resolved, run_config=config.run, store=store, progress=progress)
