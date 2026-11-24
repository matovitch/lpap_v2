import torch

from lpap.data import EnergyBank
from lpap.geometry import PairGeometry
from lpap.runtime.config import load, load_model
from lpap.runtime.hub import ArtifactStore
from lpap.stages.decoder import DecoderStageConfig, load_decoder, run_name, train


def test_decoder_stage_trains_resumes_and_exports(tmp_path):
    model = load_model({"decoder.hidden_dim": 16, "decoder.layer_count": 1, "decoder.head_count": 2})
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

    result = train(model, load(DecoderStageConfig, "decoder", overrides), geometry, bank, store)
    assert (result.step, result.reason) == (4, "target")
    assert {"probe_ce", "accuracy", "weighted_accuracy"} <= set(result.last_validation)
    assert (tmp_path / "runs" / run_name("c128_k16") / "best.pt").exists()

    longer = load(DecoderStageConfig, "decoder", {**overrides, "run.steps": 6})
    assert train(model, longer, geometry, bank, store).step == 6
    decoder = load_decoder(model, geometry, store, "cpu")
    assert decoder(
        torch.ones(2, 128), torch.nn.functional.one_hot(torch.zeros(2, 128, dtype=torch.long), 16).float()
    ).shape == (
        2,
        128,
        8,
    )
