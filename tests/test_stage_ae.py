import torch

from lpap.data import ImageSet
from lpap.geometry import PairGeometry
from lpap.runtime.config import load, load_model
from lpap.runtime.hub import ArtifactStore
from lpap.stages import ae, decoder, flow, surrogate

TINY = {
    "flow.width": 8,
    "flow.time_dim": 8,
    "flow.dilation_cycles": 1,
    "flow.dilations": [1, 2],
    "surrogate.hidden_dim": 16,
    "surrogate.layer_count": 1,
    "surrogate.head_count": 2,
    "decoder.hidden_dim": 16,
    "decoder.layer_count": 1,
    "decoder.head_count": 2,
}
FAST = {"compile": False, "bf16": False, "run.upload_every": 100}


def _teachers(model, geometries, images, store):
    flow.train(
        model,
        load(
            flow.FlowStageConfig,
            "flow",
            {
                **FAST,
                "batch_size": 4,
                "warmup_steps": 1,
                "round_trip_batch": 2,
                "validation_batch_size": 4,
                "validation_batch_count": 1,
                "gallery_count": 2,
                "round_trip_steps": [2],
                "round_trip_train_steps": 2,
                "gallery_sample_steps": [2],
                "run.steps": 1,
                "run.log_every": 1,
                "run.validate_every": 1,
                "run.checkpoint_every": 1,
            },
        ),
        images,
        store,
    )
    bank = torch.randn(40, model.value_count)
    from lpap.data import EnergyBank

    energies = EnergyBank(bank, "cpu")
    common = {
        **FAST,
        "batch_size": 4,
        "warmup_steps": 1,
        "validation_batch_size": 4,
        "validation_batch_count": 1,
        "run.steps": 1,
        "run.log_every": 1,
        "run.validate_every": 1,
        "run.checkpoint_every": 1,
    }
    for name in ("c512_k32",):
        surrogate.train(
            model, load(surrogate.SurrogateStageConfig, "surrogate", common), geometries[name], energies, store
        )
        decoder.train(model, load(decoder.DecoderStageConfig, "decoder", common), geometries[name], energies, store)


def test_ae_stage_trains_with_routed_gradients(tmp_path):
    model = load_model(TINY)
    geometries = {name: PairGeometry(model.pair(name), model.value_count) for name in ("c512_k32", "lossless")}
    images = ImageSet(torch.randint(0, 256, (40, 1, 32, 32), dtype=torch.uint8), "cpu")
    store = ArtifactStore(tmp_path, None)
    _teachers(model, geometries, images, store)
    overrides = {
        **FAST,
        "batch_size": 4,
        "warmup_steps": 1,
        "integration_steps": 2,
        "validation_batch_size": 4,
        "validation_batch_count": 1,
        "gallery_count": 2,
        "run.steps": 2,
        "run.log_every": 1,
        "run.validate_every": 2,
        "run.checkpoint_every": 2,
    }
    config = load(ae.AEStageConfig, "ae", overrides)

    task = ae.AETask(model, config, geometries, images, store)
    _, metrics, _, _ = task._forward(images.sample_train(4, torch.Generator().manual_seed(0)), train=True)
    metrics["loss"].backward()
    grads = {
        name: any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        for name, module in task.modules.items()
    }
    assert grads == {"encoder": True, "renderer": True, "surrogates": True, "decoders": True}

    # The surrogate is trained by its CE only: the image loss alone leaves its params untouched.
    task.optimizer.zero_grad(set_to_none=True)
    _, metrics, _, _ = task._forward(images.sample_train(4, torch.Generator().manual_seed(0)), train=True)
    metrics["image_l2"].backward()
    assert all(p.grad is None for p in task.surrogates.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in task.encoder.parameters())

    # Measured routing matches §3.6; side terms reach the encoder and nothing they must not.
    norms = task.gradient_norms(images.sample_train(4, torch.Generator().manual_seed(0)))
    reached = {
        term: {key.rsplit("_", 1)[1] for key in norms if key.rsplit("_", 1)[0] == f"grad_{term}"}
        for term in (
            "image_l2",
            "surrogate_ce",
            "probe_ce",
            "signed_mass",
            "residual_share",
            "usage_kl",
            "probe_ce_live",
        )
    }
    assert reached == {
        "image_l2": {"encoder", "renderer", "decoders"},
        "surrogate_ce": {"surrogates"},
        "probe_ce": {"decoders"},
        "signed_mass": {"encoder"},
        "residual_share": {"encoder"},
        "usage_kl": {"encoder"},
        "probe_ce_live": {"encoder", "decoders"},
    }

    # Fresh encoder = stage-1 flow, so the exploration metric starts at 1.
    validation = task.validate()
    assert abs(validation["stage1_corr"] - 1.0) < 1e-4
    assert abs(validation["gather_area"] - validation["gather_area_stage1"]) < 1e-3

    # Exploration noise changes what LPAP and the renderer see, not the energy itself.
    sample = images.sample_train(4, torch.Generator().manual_seed(0))
    with torch.no_grad():
        clean_energy, clean, _, _ = task._forward(sample, train=True)
        noisy_energy, noisy, _, _ = task._forward(sample, train=True, noise=0.5)
    assert torch.equal(clean_energy, noisy_energy)
    assert not torch.equal(clean["image_l2_lossless"], noisy["image_l2_lossless"])

    result = ae.train(model, config, geometries, images, store)
    assert (result.step, result.reason) == (2, "target")
    assert {"image_l2", "image_l2_lossless", "kept_share_c512_k32", "probe_accuracy_c512_k32"} <= set(
        result.last_validation
    )
    assert (tmp_path / "runs" / ae.RUN / "best.pt").exists()


def test_enabled_side_terms_train_the_encoder_only(tmp_path):
    model = load_model(TINY)
    geometries = {name: PairGeometry(model.pair(name), model.value_count) for name in ("c512_k32", "lossless")}
    images = ImageSet(torch.randint(0, 256, (40, 1, 32, 32), dtype=torch.uint8), "cpu")
    store = ArtifactStore(tmp_path, None)
    _teachers(model, geometries, images, store)
    side = {"side.residual_share": 1.0, "side.probe_ce_live": 1.0, "side.usage_kl": 1.0}
    config = load(ae.AEStageConfig, "ae", {**FAST, "integration_steps": 2, **side})
    task = ae.AETask(model, config, geometries, images, store)
    _, _, _, terms = task._forward(images.sample_train(4, torch.Generator().manual_seed(0)), train=True)
    task._side_loss(terms).backward()
    touched = {name for name, module in task.modules.items() if any(p.grad is not None for p in module.parameters())}
    assert touched == {"encoder"}


def test_gather_footprint_of_known_encoders():
    image = torch.rand(2, 1, 64)
    area, span = ae.gather_footprint(lambda x: 2.0 * x.squeeze(1), image, sites=4)
    assert torch.allclose(area, torch.tensor(1.0)) and torch.allclose(span, torch.tensor(0.0))
    box = torch.ones(1, 1, 3)
    area, span = ae.gather_footprint(lambda x: torch.nn.functional.conv1d(x, box, padding=1).squeeze(1), image, sites=4)
    assert torch.allclose(area, torch.tensor(3.0)) and torch.allclose(span, torch.tensor((2 / 3) ** 0.5))


def test_add_relative_noise_keeps_rms_gradient_and_handles_zero():
    torch.manual_seed(0)
    x = torch.ones(8, 1024, requires_grad=True)
    loss = ae.add_relative_noise(x, 1.0).square().mean()
    (grad,) = torch.autograd.grad(loss, x)
    # E[‖x + σ·rms(x)·ε‖² / N] = (1 + σ²) · mean(x²), so ∂/∂x · x ≈ 2(1 + σ²) = 4 (detached rms gives 2).
    assert abs((grad * x).sum().item() - 4.0) < 0.2

    zeros = torch.zeros(2, 64, requires_grad=True)
    (zero_grad,) = torch.autograd.grad(ae.add_relative_noise(zeros, 1.0).square().mean(), zeros)
    assert torch.isfinite(zero_grad).all()
