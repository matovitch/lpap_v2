import torch

from lpap.data import ImageSet
from lpap.runtime.config import load, load_model
from lpap.runtime.hub import ArtifactStore
from lpap.stages.flow import FlowStageConfig, load_field, train

TINY_MODEL = {"flow.width": 8, "flow.time_dim": 8, "flow.dilation_cycles": 1, "flow.dilations": [1, 2]}


def test_flow_stage_trains_resumes_and_exports(tmp_path):
    model = load_model(TINY_MODEL)
    overrides = {
        "batch_size": 4,
        "validation_batch_size": 4,
        "validation_batch_count": 2,
        "round_trip_steps": [1, 2],
        "gallery_count": 3,
        "gallery_sample_steps": [1, 2],
        "warmup_steps": 2,
        "compile": False,
        "round_trip_weight": 0.5,
        "round_trip_batch": 2,
        "round_trip_bf16": True,
        "round_trip_every": 2,
        "bf16": False,
        "run.steps": 6,
        "run.log_every": 1,
        "run.validate_every": 3,
        "run.checkpoint_every": 3,
        "run.upload_every": 100,
    }
    config = load(FlowStageConfig, "flow", overrides)
    images = ImageSet(torch.randint(0, 256, (200, 1, 32, 32), dtype=torch.uint8), "cpu")
    store = ArtifactStore(tmp_path, None)

    result = train(model, config, images, store)
    assert (result.step, result.reason) == (6, "target")
    assert {
        "flow_matching",
        "fm_i2e_a0.5",
        "round_trip_l2_2",
        "round_trip_l2_4",
        "objective",
        "energy_top10",
        "prior_top10",
    } <= set(result.last_validation)
    assert (tmp_path / "runs" / "flow" / "gallery.png").read_bytes().startswith(b"\x89PNG")

    longer = load(FlowStageConfig, "flow", {**overrides, "run.steps": 9})
    assert train(model, longer, images, store).step == 9
    assert load_field(model, store, "cpu").output.weight.shape == (1, 8, 1)
