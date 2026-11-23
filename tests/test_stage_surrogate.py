import torch

from lpap.data import EnergyBank
from lpap.geometry import PairGeometry
from lpap.runtime.config import load, load_model
from lpap.runtime.hub import ArtifactStore
from lpap.stages.surrogate import SurrogateStageConfig, load_surrogate, run_name, train

TINY_MODEL = {"surrogate.hidden_dim": 16, "surrogate.layer_count": 1, "surrogate.head_count": 2}


def test_surrogate_stage_trains_resumes_and_exports(tmp_path):
    model = load_model(TINY_MODEL)
    geometry = PairGeometry(model.pair("c128_k16"), model.value_count)
    bank = EnergyBank(torch.randn(120, model.value_count), "cpu")
    store = ArtifactStore(tmp_path, None)
    overrides = {
        "batch_size": 8,
        "warmup_steps": 2,
        "compile": False,
        "bf16": False,
        "validation_batch_size": 4,
        "validation_batch_count": 1,
        "run.steps": 4,
        "run.log_every": 1,
        "run.validate_every": 2,
        "run.checkpoint_every": 2,
        "run.upload_every": 100,
    }

    result = train(model, load(SurrogateStageConfig, "surrogate", overrides), geometry, bank, store)
    assert (result.step, result.reason) == (4, "target")
    assert {"surrogate_ce", "accuracy", "weighted_accuracy"} <= set(result.last_validation)
    assert (tmp_path / "runs" / run_name("c128_k16") / "best.pt").exists()

    longer = load(SurrogateStageConfig, "surrogate", {**overrides, "run.steps": 6})
    assert train(model, longer, geometry, bank, store).step == 6
    surrogate = load_surrogate(model, geometry, store, "cpu")
    assert surrogate(geometry.fold(bank.batch(torch.arange(2)))).shape == (2, 128, geometry.class_count)
