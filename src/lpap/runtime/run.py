"""Foreground training loop: step target, wall-clock budget, interrupt = save (FOUNDATION §7).

No threads, no subprocesses. The loop runs inside the notebook cell that
called it. A kernel interrupt is caught: the checkpoint is saved and
uploaded, then the call returns normally. Re-running the cell resumes.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from lpap.runtime import checkpoint, notify
from lpap.runtime.hub import ArtifactStore
from lpap.runtime.log import MetricLog

LAST = "last.pt"
BEST = "best.pt"
LOG = "log.sqlite"
STATUS = "status.json"
GALLERY = "gallery.png"


@dataclass(frozen=True)
class RunConfig:
    steps: int
    budget_minutes: float
    log_every: int
    validate_every: int
    checkpoint_every: int
    upload_every: int


class Task(Protocol):
    """One stage's trainable state. ``train_step`` metrics may stay on device.

    Optional: ``gallery(path)`` writes a review image after each validation.
    """

    best_metric: str

    def train_step(self) -> Mapping[str, torch.Tensor | float]: ...

    def validate(self) -> Mapping[str, float]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...

    def weights(self) -> dict[str, Any]: ...


@dataclass
class RunResult:
    run: str
    step: int
    reason: str
    best: float
    elapsed_minutes: float
    last_train: dict[str, float] = field(default_factory=dict)
    last_validation: dict[str, float] = field(default_factory=dict)


ProgressFn = Callable[[dict[str, Any]], None]


def run(
    task: Task,
    *,
    run: str,
    config: Mapping[str, Any],
    run_config: RunConfig,
    store: ArtifactStore,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Train ``task`` until the step target, the budget or an interrupt."""
    started = time.monotonic()
    step, best = 0, math.inf
    last_path = store.fetch(run, LAST)
    if last_path is not None:
        state = checkpoint.load(last_path)
        task.load_state_dict(state["task"])
        checkpoint.set_rng_state(state["rng"])
        step, best = state["step"], state["best"]
        store.fetch(run, LOG)
    log = MetricLog(store.local(run, LOG), start_step=step, config=config)
    status = _Status(store.root / STATUS, run=run, target=run_config.steps)

    def save_last() -> None:
        payload = {
            "task": task.state_dict(),
            "rng": checkpoint.rng_state(),
            "step": step,
            "best": best,
            "config": dict(config),
        }
        checkpoint.save(payload, store.local(run, LAST))

    def save_best(value: float) -> None:
        payload = {"weights": task.weights(), "step": step, "value": value, "config": dict(config)}
        checkpoint.save(payload, store.local(run, BEST))

    reason = "target"
    gallery = getattr(task, "gallery", None)
    # Per-metric counts: tasks may report a metric only on some steps.
    window: dict[str, torch.Tensor | float] = {}
    counts: dict[str, int] = {}
    train_metrics: dict[str, float] = {}
    validation: dict[str, float] = {}
    best_dirty = False
    try:
        status.write("running", step, {})
        while step < run_config.steps:
            if time.monotonic() - started > 60.0 * run_config.budget_minutes:
                reason = "budget"
                break
            metrics = task.train_step()
            step += 1
            for name, value in metrics.items():
                window[name] = window.get(name, 0.0) + (value.detach() if torch.is_tensor(value) else value)
                counts[name] = counts.get(name, 0) + 1
            if step % run_config.log_every == 0:
                train_metrics = {name: float(value) / counts[name] for name, value in window.items()}
                window, counts = {}, {}
                if not all(math.isfinite(value) for value in train_metrics.values()):
                    raise FloatingPointError(f"non-finite training metrics at step {step}: {train_metrics}")
                log.write(step, "train", train_metrics)
                status.write("running", step, train_metrics)
                if progress:
                    progress({"step": step, "target": run_config.steps, "train": train_metrics, "best": best})
            if step % run_config.validate_every == 0 or step == run_config.steps:
                validation = dict(task.validate())
                log.write(step, "validation", validation)
                if validation[task.best_metric] < best:
                    best = validation[task.best_metric]
                    save_best(best)
                    best_dirty = True
                event = {"step": step, "target": run_config.steps, "validation": validation, "best": best}
                if gallery is not None:
                    gallery(store.local(run, GALLERY))
                    event["gallery"] = store.local(run, GALLERY)
                if progress:
                    progress(event)
            if step % run_config.checkpoint_every == 0:
                save_last()
            if step % run_config.upload_every == 0:
                _upload(store, run, step, best_dirty)
                best_dirty = False
    except KeyboardInterrupt:
        reason = "interrupt"
    except BaseException as error:
        log.close()
        status.write("failed", step, train_metrics)
        notify.send(f"lpap {run} failed", f"step {step}: {type(error).__name__}: {error}", priority=1)
        raise
    save_last()
    log.close()
    _upload(store, run, step, best_dirty)
    elapsed = (time.monotonic() - started) / 60.0
    status.write(reason, step, train_metrics)
    notify.send(f"lpap {run}: {reason}", f"step {step}/{run_config.steps}, best {best:.5g}, {elapsed:.0f} min")
    return RunResult(run, step, reason, best, elapsed, train_metrics, validation)


def _upload(store: ArtifactStore, run: str, step: int, best_dirty: bool) -> None:
    files = [LAST, LOG, GALLERY] + ([BEST] if best_dirty else [])
    store.push(run, files, f"{run}: step {step}")


GPU_QUERY = "utilization.gpu,power.draw,power.limit,memory.used,memory.total"
GPU_EVERY_SECONDS = 10.0


def gpu_stats() -> dict[str, float]:
    """First GPU's ``nvidia-smi`` readings (utilization %, watts, MiB); empty when unavailable."""
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.splitlines()[0]
        return {key: float(value) for key, value in zip(GPU_QUERY.split(","), line.split(","), strict=True)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {}


class _Status:
    """Heartbeat file, readable through the molab file API without the kernel."""

    def __init__(self, path: Path, *, run: str, target: int) -> None:
        self.path = path
        self.run = run
        self.target = target
        self.started = time.time()
        self.gpu: dict[str, float] = {}
        self.gpu_sampled = -math.inf

    def write(self, state: str, step: int, metrics: Mapping[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Throttled: nvidia-smi takes tens of milliseconds.
        if time.monotonic() - self.gpu_sampled > GPU_EVERY_SECONDS:
            self.gpu, self.gpu_sampled = gpu_stats(), time.monotonic()
        payload = {
            "gpu": self.gpu,
            "state": state,
            "run": self.run,
            "step": step,
            "target": self.target,
            "metrics": dict(metrics),
            "started": self.started,
            "heartbeat": time.time(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)
