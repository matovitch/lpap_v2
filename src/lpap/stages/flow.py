"""Stage 1: train the bidirectional flow on images and prior draws (FOUNDATION §5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lpap.data import ImageSet
from lpap.flow import FlowField, encode, render, sample_prior
from lpap.losses import flow_matching, sample_bridge_time
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
    max_grad_norm: float
    time_eps: float
    validation_seed: int
    validation_batch_size: int
    validation_batch_count: int
    euler_steps: int
    run: RunConfig


class FlowTask:
    best_metric = "flow_matching"

    def __init__(self, model: ModelConfig, config: FlowStageConfig, images: ImageSet) -> None:
        self.model = model
        self.config = config
        self.images = images
        self.device = images.device
        torch.manual_seed(config.seed)
        self.field = FlowField(model.flow).to(self.device)
        self.optimizer = torch.optim.AdamW(self.field.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.validation_images = images.validation_batches(config.validation_batch_size, config.validation_batch_count)

    def _prior(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        shape = (batch, 1, self.model.value_count)
        prior = self.model.prior
        return sample_prior(shape, sigma=prior.sigma, scale=prior.scale, generator=generator).to(self.device)

    def _time(self, batch: int, generator: torch.Generator) -> torch.Tensor:
        return sample_bridge_time(batch, eps=self.config.time_eps, generator=generator, device=self.device)

    def train_step(self) -> dict[str, torch.Tensor]:
        self.field.train()
        image = self.images.sample_train(self.config.batch_size, self.generator)
        prior = self._prior(image.shape[0], self.generator)
        loss = flow_matching(self.field, image, prior, self._time(image.shape[0], self.generator))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.field.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return {"flow_matching": loss.detach(), "grad_norm": grad_norm}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Fixed images, prior draws and times: comparable across steps and runs."""
        self.field.eval()
        generator = torch.Generator().manual_seed(self.config.validation_seed)
        totals: dict[str, float] = {}
        for image in self.validation_images:
            batch = image.shape[0]
            prior = self._prior(batch, generator)
            energy = encode(self.field, image, self.config.euler_steps)
            metrics = {
                "flow_matching": flow_matching(self.field, image, prior, self._time(batch, generator)),
                "round_trip_l2": (render(self.field, energy, self.config.euler_steps) - image).square().mean(),
                "energy_std": energy.std(),
                "energy_abs_mean": energy.abs().mean(),
            }
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) / len(self.validation_images)
        return totals

    def state_dict(self) -> dict[str, Any]:
        return {
            "field": self.field.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.field.load_state_dict(state["field"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])

    def weights(self) -> dict[str, Any]:
        return {"field": self.field.state_dict()}


def train(
    model: ModelConfig,
    config: FlowStageConfig,
    images: ImageSet,
    store: ArtifactStore,
    progress: ProgressFn | None = None,
) -> RunResult:
    task = FlowTask(model, config, images)
    resolved = {"model": to_dict(model), "stage": to_dict(config)}
    return run(task, run=RUN, config=resolved, run_config=config.run, store=store, progress=progress)


def load_field(model: ModelConfig, store: ArtifactStore, device: torch.device | str) -> FlowField:
    """Best flow weights, for the stages that follow."""
    path = store.fetch(RUN, BEST)
    if path is None:
        raise FileNotFoundError("no flow best checkpoint; run stage 1 first")
    field = FlowField(model.flow)
    field.load_state_dict(checkpoint.load(path)["weights"]["field"])
    return field.to(device)
