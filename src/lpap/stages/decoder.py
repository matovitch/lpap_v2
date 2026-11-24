"""Stage 4: one probe decoder per lossy pair (FOUNDATION §2.6, §4.3, §5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lpap.codec import DecoderInputs, frontend
from lpap.data import EnergyBank
from lpap.decoder import Decoder
from lpap.geometry import PairGeometry
from lpap.losses import accuracies, probe_ce
from lpap.lpap import ExactCode, exact_code
from lpap.runtime import checkpoint
from lpap.runtime.config import ModelConfig, to_dict
from lpap.runtime.hub import ArtifactStore
from lpap.runtime.run import BEST, ProgressFn, RunConfig, RunResult, run
from lpap.stages.flow import learning_rate


def run_name(pair: str) -> str:
    return f"decoder-{pair}"


@dataclass(frozen=True)
class DecoderStageConfig:
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


class DecoderTask:
    best_metric = "probe_ce"

    def __init__(
        self, model: ModelConfig, config: DecoderStageConfig, geometry: PairGeometry, bank: EnergyBank
    ) -> None:
        self.config = config
        self.geometry = geometry
        self.bank = bank
        self.device = bank.device
        torch.manual_seed(config.seed)
        self.decoder = Decoder(geometry, model.decoder).to(self.device)
        self.train_decoder = torch.compile(self.decoder) if config.compile else self.decoder
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=config.learning_rate)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step = 0
        self.validation_energies = bank.validation_batches(config.validation_batch_size, config.validation_batch_count)

    def _inputs(self, energy: torch.Tensor) -> tuple[DecoderInputs, ExactCode]:
        code = exact_code(energy, self.geometry)
        inputs = frontend(self.geometry.fold(energy), code, self.geometry)
        return inputs, code

    def train_step(self) -> dict[str, torch.Tensor | float]:
        self.decoder.train()
        rate = learning_rate(self.config, self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        energy = self.bank.sample_train(self.config.batch_size, self.generator)
        inputs, code = self._inputs(energy)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.config.bf16):
            logits = self.train_decoder(inputs.amplitudes, inputs.dib_weights)
        loss = probe_ce(logits, code)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.step += 1
        accuracy, weighted = accuracies(logits, code.probes, code.amplitudes.abs())
        return {
            "probe_ce": loss.detach(),
            "accuracy": accuracy,
            "weighted_accuracy": weighted,
            "grad_norm": grad_norm,
            "lr": rate,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.decoder.eval()
        totals: dict[str, float] = {}
        for energy in self.validation_energies:
            inputs, code = self._inputs(energy)
            logits = self.decoder(inputs.amplitudes, inputs.dib_weights)
            accuracy, weighted = accuracies(logits, code.probes, code.amplitudes.abs())
            metrics = {"probe_ce": probe_ce(logits, code), "accuracy": accuracy, "weighted_accuracy": weighted}
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) / len(self.validation_energies)
        return totals

    def state_dict(self) -> dict[str, Any]:
        return {
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "step": self.step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decoder.load_state_dict(state["decoder"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.step = state["step"]

    def weights(self) -> dict[str, Any]:
        return {"decoder": self.decoder.state_dict()}


def train(
    model: ModelConfig,
    config: DecoderStageConfig,
    geometry: PairGeometry,
    bank: EnergyBank,
    store: ArtifactStore,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Train the decoder of one lossy pair into run ``decoder-<pair>``."""
    task = DecoderTask(model, config, geometry, bank)
    resolved = {"model": to_dict(model), "stage": to_dict(config), "pair": geometry.name}
    return run(
        task, run=run_name(geometry.name), config=resolved, run_config=config.run, store=store, progress=progress
    )


def load_decoder(
    model: ModelConfig, geometry: PairGeometry, store: ArtifactStore, device: torch.device | str
) -> Decoder:
    """Best decoder of one pair, for stage 5."""
    path = store.fetch(run_name(geometry.name), BEST)
    if path is None:
        raise FileNotFoundError(f"no decoder best checkpoint for {geometry.name}; run stage 4 first")
    decoder = Decoder(geometry, model.decoder)
    decoder.load_state_dict(checkpoint.load(path)["weights"]["decoder"])
    return decoder.to(device)
