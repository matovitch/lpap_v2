import math

import torch

from lpap.data import ImageSet
from lpap.flow import encode
from lpap.runtime.config import load, load_model
from lpap.runtime.hub import ArtifactStore
from lpap.stages import bank as bank_stage
from lpap.stages.flow import FlowStageConfig, load_field, train

TINY_MODEL = {"flow.width": 8, "flow.time_dim": 8, "flow.dilation_cycles": 1, "flow.dilations": [1, 2]}


def test_bank_encodes_every_image_in_order(tmp_path):
    model = load_model(TINY_MODEL)
    flow_overrides = {
        "batch_size": 4,
        "validation_batch_size": 4,
        "validation_batch_count": 1,
        "round_trip_steps": [1, 2],
        "gallery_count": 2,
        "gallery_sample_steps": [1],
        "warmup_steps": 1,
        "compile": False,
        "run.steps": 2,
        "run.validate_every": 2,
        "run.checkpoint_every": 2,
        "run.upload_every": 100,
    }
    images = ImageSet(torch.randint(0, 256, (150, 1, 32, 32), dtype=torch.uint8), "cpu")
    store = ArtifactStore(tmp_path, None)
    train(model, load(FlowStageConfig, "flow", flow_overrides), images, store)

    # 150 = 2 full chunks + a partial one; progress reaches the total.
    config = load(bank_stage.BankStageConfig, "bank", {"batch_size": 64, "compile": False, "check_count": 8})
    seen = []
    bank, stats = bank_stage.build(model, config, images, store, lambda done, total: seen.append((done, total)))

    assert bank.energies.shape == (150, 1024) and bank.energies.dtype == torch.float16
    assert seen[-1] == (150, 150)
    expected = encode(load_field(model, store, "cpu"), images.batch(torch.arange(150)), config.steps)
    torch.testing.assert_close(bank.batch(torch.arange(150)), expected, atol=1e-2, rtol=1e-2)
    assert torch.equal(bank.validation, images.validation)
    assert all(math.isfinite(value) for value in stats.values())
