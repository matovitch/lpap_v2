"""Stage 3: one surrogate per lossy pair, trained by CE on energy-bank rows (FOUNDATION §4.2, §5).

Inputs are fixed bank energies, so the CE trains the surrogate only (there is
no encoder in this stage). Targets are exact LPAP local classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from lpap.data import EnergyBank
from lpap.geometry import PairGeometry
from lpap.losses import accuracies, surrogate_ce
from lpap.lpap import exact_code
from lpap.runtime import checkpoint
from lpap.runtime.config import ModelConfig, to_dict
from lpap.runtime.hub import ArtifactStore
from lpap.runtime.run import BEST, ProgressFn, RunConfig, RunResult, run
from lpap.stages.flow import learning_rate
from lpap.surrogate import Surrogate


def run_name(pair: str) -> str:
    return f"surrogate-{pair}"


@dataclass(frozen=True)
class SurrogateStageConfig:
    seed: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    min_lr_ratio: float
    max_grad_norm: float
    compile: bool
    bf16: bool
    validation_batch_size: int
    validation_batch_count: int
    run: RunConfig


class SurrogateTask:
    best_metric = "surrogate_ce"

    def __init__(
        self, model: ModelConfig, config: SurrogateStageConfig, geometry: PairGeometry, bank: EnergyBank
    ) -> None:
        self.config = config
        self.geometry = geometry
        self.bank = bank
        self.device = bank.device
        torch.manual_seed(config.seed)
        self.surrogate = Surrogate(geometry, model.surrogate).to(self.device)
        self.train_surrogate = torch.compile(self.surrogate) if config.compile else self.surrogate
        self.optimizer = torch.optim.AdamW(self.surrogate.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step = 0
        self.validation_energies = bank.validation_batches(config.validation_batch_size, config.validation_batch_count)

    def train_step(self) -> dict[str, torch.Tensor]:
        self.surrogate.train()
        rate = learning_rate(self.config, self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        energy = self.bank.sample_train(self.config.batch_size, self.generator)
        code = exact_code(energy, self.geometry)
        tokens = self.geometry.fold(energy)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.config.bf16):
            logits = self.train_surrogate(tokens)
        loss = surrogate_ce(logits, code)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.surrogate.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.step += 1
        accuracy, weighted = accuracies(logits, code.local_classes, code.amplitudes.abs())
        return {
            "surrogate_ce": loss.detach(),
            "accuracy": accuracy,
            "weighted_accuracy": weighted,
            "grad_norm": grad_norm,
            "lr": rate,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Fixed validation bank rows, fp32 eager forward."""
        self.surrogate.eval()
        totals: dict[str, float] = {}
        for energy in self.validation_energies:
            code = exact_code(energy, self.geometry)
            logits = self.surrogate(self.geometry.fold(energy))
            accuracy, weighted = accuracies(logits, code.local_classes, code.amplitudes.abs())
            metrics = {"surrogate_ce": surrogate_ce(logits, code), "accuracy": accuracy, "weighted_accuracy": weighted}
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) / len(self.validation_energies)
        return totals

    def state_dict(self) -> dict[str, Any]:
        return {
            "surrogate": self.surrogate.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.surrogate.load_state_dict(state["surrogate"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.step = state["step"]

    def weights(self) -> dict[str, Any]:
        return {"surrogate": self.surrogate.state_dict()}


def train(
    model: ModelConfig,
    config: SurrogateStageConfig,
    geometry: PairGeometry,
    bank: EnergyBank,
    store: ArtifactStore,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Train the surrogate of one lossy pair into run ``surrogate-<pair>``."""
    task = SurrogateTask(model, config, geometry, bank)
    resolved = {"model": to_dict(model), "stage": to_dict(config), "pair": geometry.name}
    return run(
        task, run=run_name(geometry.name), config=resolved, run_config=config.run, store=store, progress=progress
    )


def load_surrogate(
    model: ModelConfig, geometry: PairGeometry, store: ArtifactStore, device: torch.device | str
) -> Surrogate:
    """Best surrogate of one pair, for stage 5."""
    path = store.fetch(run_name(geometry.name), BEST)
    if path is None:
        raise FileNotFoundError(f"no surrogate best checkpoint for {geometry.name}; run stage 3 first")
    surrogate = Surrogate(geometry, model.surrogate)
    surrogate.load_state_dict(checkpoint.load(path)["weights"]["surrogate"])
    return surrogate.to(device)
