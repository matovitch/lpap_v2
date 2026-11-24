import json

import pytest
import torch

from lpap.runtime import checkpoint
from lpap.runtime.hub import ArtifactStore
from lpap.runtime.log import read_series
from lpap.runtime.run import BEST, LAST, LOG, STATUS, RunConfig, run


class CountingTask:
    """Loss = 1 / (1 + weight); each step adds 1 to the weight."""

    best_metric = "loss"

    def __init__(self, interrupt_at=None, fail_at=None):
        self.weight = torch.zeros(())
        self.interrupt_at = interrupt_at
        self.fail_at = fail_at
        self.steps = 0

    def train_step(self):
        self.steps += 1
        if self.steps == self.interrupt_at:
            raise KeyboardInterrupt
        if self.steps == self.fail_at:
            return {"loss": torch.tensor(float("nan"))}
        self.weight += 1
        sparse = {"even": float(self.weight)} if self.steps % 2 == 0 else {}
        return {"loss": 1.0 / (1.0 + self.weight), **sparse}

    def validate(self):
        return {"loss": float(1.0 / (1.0 + self.weight))}

    def state_dict(self):
        return {"weight": self.weight.clone()}

    def load_state_dict(self, state):
        self.weight = state["weight"].clone()

    def weights(self):
        return {"weight": self.weight.clone()}


def config(steps=20, **changes):
    values = dict(steps=steps, budget_minutes=60.0, log_every=1, validate_every=5, checkpoint_every=5, upload_every=10)
    return RunConfig(**{**values, **changes})


def test_runs_to_target_and_writes_artifacts(tmp_path):
    store = ArtifactStore(tmp_path, None)
    seen = []
    result = run(CountingTask(), run="r", config={"a": 1}, run_config=config(), store=store, progress=seen.append)
    assert (result.step, result.reason) == (20, "target")
    last = checkpoint.load(store.local("r", LAST))
    assert last["step"] == 20 and float(last["task"]["weight"]) == 20.0
    assert checkpoint.load(store.local("r", BEST))["step"] == 20
    assert [step for step, _ in read_series(store.local("r", LOG), "validation", "loss")] == [5, 10, 15, 20]
    assert json.loads((tmp_path / STATUS).read_text())["state"] == "target"
    assert seen[-1]["step"] == 20


def test_interrupt_saves_and_resume_continues(tmp_path):
    store = ArtifactStore(tmp_path, None)
    first = run(CountingTask(interrupt_at=8), run="r", config={}, run_config=config(), store=store)
    assert (first.step, first.reason) == (7, "interrupt")
    assert checkpoint.load(store.local("r", LAST))["step"] == 7

    task = CountingTask()
    second = run(task, run="r", config={}, run_config=config(), store=store)
    assert (second.step, second.reason) == (20, "target")
    assert float(task.weight) == 20.0 and task.steps == 13
    steps = [step for step, _ in read_series(store.local("r", LOG), "train", "loss")]
    assert steps == list(range(1, 21))


def test_resume_drops_log_rows_after_the_checkpoint(tmp_path):
    store = ArtifactStore(tmp_path, None)
    run(CountingTask(), run="r", config={}, run_config=config(steps=7, checkpoint_every=100), store=store)
    # Simulate a lost final save: roll the checkpoint back to step 5.
    state = checkpoint.load(store.local("r", LAST))
    state["step"], state["task"]["weight"] = 5, torch.tensor(5.0)
    checkpoint.save(state, store.local("r", LAST))
    run(CountingTask(), run="r", config={}, run_config=config(steps=7), store=store)
    steps = [step for step, _ in read_series(store.local("r", LOG), "train", "loss")]
    assert steps == list(range(1, 8))


def test_stop_at_pauses_and_a_later_rung_resumes(tmp_path):
    store = ArtifactStore(tmp_path, None)
    first = run(CountingTask(), run="r", config={}, run_config=config(stop_at=12), store=store)
    assert (first.step, first.reason) == (12, "stop_at")
    validated = [step for step, _ in read_series(store.local("r", LOG), "validation", "loss")]
    assert validated == [5, 10, 12]
    second = run(CountingTask(), run="r", config={}, run_config=config(), store=store)
    assert (second.step, second.reason) == (20, "target")


def test_budget_stops_the_run(tmp_path):
    result = run(
        CountingTask(), run="r", config={}, run_config=config(budget_minutes=0.0), store=ArtifactStore(tmp_path, None)
    )
    assert (result.step, result.reason) == (0, "budget")


def test_non_finite_metrics_fail_without_overwriting_last(tmp_path):
    store = ArtifactStore(tmp_path, None)
    with pytest.raises(FloatingPointError):
        run(CountingTask(fail_at=7), run="r", config={}, run_config=config(), store=store)
    assert checkpoint.load(store.local("r", LAST))["step"] == 5
    assert json.loads((tmp_path / STATUS).read_text())["state"] == "failed"


def test_train_metrics_are_averaged_over_the_log_window(tmp_path):
    store = ArtifactStore(tmp_path, None)
    run(CountingTask(), run="r", config={}, run_config=config(steps=4, log_every=2), store=store)
    logged = [value for _, value in read_series(store.local("r", LOG), "train", "loss")]
    assert logged == pytest.approx([(1 / 2 + 1 / 3) / 2, (1 / 4 + 1 / 5) / 2])
    # Metrics reported on some steps only are averaged over those steps.
    assert [value for _, value in read_series(store.local("r", LOG), "train", "even")] == [2.0, 4.0]
