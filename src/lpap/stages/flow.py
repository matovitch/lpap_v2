"""Stage 1: train the bidirectional flow on images and prior draws (FOUNDATION §5)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lpap.data import ImageSet
from lpap.flow import FlowField, encode, render, sample_prior
from lpap.gallery import energy_tiles, grid, image_tiles, write_png
from lpap.losses import flow_matching, round_trip_l2, sample_bridge_time
from lpap.runtime import checkpoint
from lpap.runtime.config import ModelConfig, to_dict
from lpap.runtime.hub import ArtifactStore
from lpap.runtime.run import BEST, ProgressFn, RunConfig, RunResult, run

RUN = "flow"


@dataclass(frozen=True)
class FlowStageConfig:
    seed: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    min_lr_ratio: float
    ema_decay: float
    max_grad_norm: float
    compile: bool
    bf16: bool
    time_eps: float
    time_concentration: float
    round_trip_weight: float
    round_trip_batch: int
    round_trip_train_steps: int
    round_trip_bf16: bool
    round_trip_every: int
    validation_seed: int
    validation_batch_size: int
    validation_batch_count: int
    round_trip_steps: tuple[int, ...]
    band_alphas: tuple[float, ...]
    gallery_count: int
    gallery_sample_steps: tuple[int, ...]
    run: RunConfig


def learning_rate(config: Any, step: int) -> float:
    """Linear warmup, then cosine down to ``min_lr_ratio`` at the step target.

    ``config``: any stage config with ``learning_rate``, ``warmup_steps``, ``min_lr_ratio`` and ``run.steps``.
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    span = max(config.run.steps - config.warmup_steps, 1)
    progress = min((step - config.warmup_steps) / span, 1.0)
    ratio = config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * ratio


def top_share(values: torch.Tensor, fraction: float) -> torch.Tensor:
    """Mean share of each row's squared mass held by its largest ``fraction`` of coefficients."""
    square = values.flatten(1).float().square()
    count = max(1, round(fraction * square.shape[1]))
    return (square.topk(count, dim=1).values.sum(1) / square.sum(1).clamp_min(1e-12)).mean()


class FlowTask:
    """Trains ``field``; validation, gallery and exported weights use the EMA copy."""

    def __init__(self, model: ModelConfig, config: FlowStageConfig, images: ImageSet) -> None:
        self.model = model
        self.config = config
        # Validation analogue of the training loss: round-trip metrics alone favour barely trained fields.
        self.best_metric = "objective"
        self.images = images
        self.device = images.device
        torch.manual_seed(config.seed)
        self.field = FlowField(model.flow).to(self.device)
        self.ema = copy.deepcopy(self.field).requires_grad_(False)
        # Compiled views share parameters with the eager modules; checkpoints use the eager ones.
        self.train_field = torch.compile(self.field) if config.compile else self.field
        self.eval_field = torch.compile(self.ema) if config.compile else self.ema
        self.optimizer = torch.optim.AdamW(self.field.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step = 0
        self.validation_images = images.validation_batches(config.validation_batch_size, config.validation_batch_count)

    def _prior(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        shape = (batch, 1, self.model.value_count)
        prior = self.model.prior
        return sample_prior(shape, sigma=prior.sigma, scale=prior.scale, generator=generator).to(self.device)

    def _time(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        config = self.config
        return sample_bridge_time(
            batch, eps=config.time_eps, concentration=config.time_concentration, generator=generator, device=self.device
        )

    def _euler_time(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        """Times the integrators query: midpoints of every ``round_trip_steps`` grid, both halves."""
        grid = torch.cat([(torch.arange(k) + 0.5) / k for k in self.config.round_trip_steps])
        local = grid[torch.randint(0, grid.numel(), (batch,), generator=generator)]
        left = torch.rand(batch, generator=generator) < 0.5
        return torch.where(left, local - 1.0, local).to(self.device)

    def train_step(self) -> dict[str, torch.Tensor]:
        self.field.train()
        rate = learning_rate(self.config, self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        image = self.images.sample_train(self.config.batch_size, self.generator)
        config = self.config
        prior = self._prior(image.shape[0], self.generator)
        time = self._time(image.shape[0], self.generator)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=config.bf16):
            loss = flow_matching(self.train_field, image, prior, time)
        metrics = {"flow_matching": loss.detach()}
        if config.round_trip_weight > 0 and self.step % config.round_trip_every == 0:
            round_trip = round_trip_l2(
                self.train_field,
                image[: config.round_trip_batch],
                config.round_trip_train_steps,
                bf16=config.round_trip_bf16,
            )
            # Scaled by the period so the term's average weight does not depend on it.
            loss = loss + config.round_trip_weight * config.round_trip_every * round_trip
            metrics["round_trip_l2"] = round_trip.detach()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.field.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.step += 1
        decay = min(self.config.ema_decay, (1.0 + self.step) / (10.0 + self.step))
        torch._foreach_lerp_(list(self.ema.parameters()), list(self.field.parameters()), 1.0 - decay)
        return {**metrics, "grad_norm": grad_norm, "lr": rate}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Fixed images, prior draws and times: comparable across steps and runs. fp32.

        ``fm_euler``: loss at the Euler midpoints the integrators query
        (``flow_matching`` also weighs ``|t| → 0``, where the halves are
        indistinguishable). ``fm_<half>_a<α>``: loss at a fixed prior weight ``α`` (bands separate the
        irreducible floor near each end from learnable error). ``round_trip_l2_<k>``:
        encode then render with ``k`` Euler steps each. ``objective`` (best metric):
        ``fm_euler + round_trip_weight · round_trip_l2_<round_trip_train_steps>``.
        """
        self.ema.eval()
        field = self.eval_field
        config = self.config
        generator = torch.Generator().manual_seed(config.validation_seed)
        totals: dict[str, float] = {}
        for image in self.validation_images:
            batch = image.shape[0]
            prior = self._prior(batch, generator)
            ones = torch.ones(batch, device=self.device)
            metrics = {
                "flow_matching": flow_matching(field, image, prior, self._time(batch, generator)),
                "fm_euler": flow_matching(field, image, prior, self._euler_time(batch, generator)),
            }
            for alpha in config.band_alphas:
                metrics[f"fm_i2e_a{alpha:g}"] = flow_matching(field, image, prior, (alpha - 1.0) * ones)
                metrics[f"fm_e2i_a{alpha:g}"] = flow_matching(field, image, prior, (1.0 - alpha) * ones)
            for steps in sorted({*config.round_trip_steps, config.round_trip_train_steps}):
                metrics[f"round_trip_l2_{steps}"] = round_trip_l2(field, image, steps)
            trained = metrics[f"round_trip_l2_{config.round_trip_train_steps}"]
            metrics["objective"] = metrics["fm_euler"] + config.round_trip_weight * trained
            # Goal of stage 1: energies as concentrated and centred as the prior (side by side).
            energy = encode(field, image, config.round_trip_train_steps)
            for name, values in (("energy", energy), ("prior", prior.squeeze(1))):
                metrics[f"{name}_mean"] = values.mean()
                metrics[f"{name}_std"] = values.std()
                metrics[f"{name}_top10"] = top_share(values, 0.1)
            sample = render(field, prior.squeeze(1), max(config.round_trip_steps))
            metrics["sample_mean"] = sample.mean()
            metrics["sample_std"] = sample.std()
            metrics["image_mean"] = image.mean()
            metrics["image_std"] = image.std()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) / len(self.validation_images)
        return totals

    @torch.no_grad()
    def gallery(self, path: Path) -> Path:
        """Rows: images, energies, round trips per step count, prior samples per step count."""
        field = self.ema.eval()
        config = self.config
        image = self.validation_images[0][: config.gallery_count]
        generator = torch.Generator().manual_seed(config.validation_seed + 1)
        prior = self._prior(image.shape[0], generator).squeeze(1)
        rows = [image_tiles(image), energy_tiles(encode(field, image, max(config.round_trip_steps)))]
        rows += [image_tiles(render(field, encode(field, image, k), k)) for k in config.round_trip_steps]
        rows += [energy_tiles(prior)]
        rows += [image_tiles(render(field, prior, k)) for k in config.gallery_sample_steps]
        return write_png(path, grid(rows))

    def state_dict(self) -> dict[str, Any]:
        return {
            "field": self.field.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.field.load_state_dict(state["field"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.step = state["step"]

    def weights(self) -> dict[str, Any]:
        return {"field": self.ema.state_dict()}


def train(
    model: ModelConfig,
    config: FlowStageConfig,
    images: ImageSet,
    store: ArtifactStore,
    progress: ProgressFn | None = None,
    *,
    name: str = RUN,
) -> RunResult:
    """Train into run ``name`` (``flow`` is what later stages load; others are experiments)."""
    task = FlowTask(model, config, images)
    resolved = {"model": to_dict(model), "stage": to_dict(config)}
    return run(task, run=name, config=resolved, run_config=config.run, store=store, progress=progress)


def load_field(model: ModelConfig, store: ArtifactStore, device: torch.device | str) -> FlowField:
    """Best flow weights, for the stages that follow."""
    path = store.fetch(RUN, BEST)
    if path is None:
        raise FileNotFoundError("no flow best checkpoint; run stage 1 first")
    field = FlowField(model.flow)
    field.load_state_dict(checkpoint.load(path)["weights"]["field"])
    return field.to(device)
