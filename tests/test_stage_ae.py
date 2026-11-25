import pytest
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


def test_noise_controller_grad_norm_and_snr():
    p = torch.nn.Parameter(torch.zeros(4))

    # grad_norm: starts at σ_max, decays when ‖g‖ drops below its peak, and resumes cleanly.
    ctrl = ae.NoiseController(ae.Explore(1.0, 0, "grad_norm", 4.0, None))
    assert ctrl.sigma(0) == 1.0
    for _ in range(12):
        p.grad = torch.ones(4)
        ctrl.observe([p])
    peak_sigma = ctrl.sigma(12)
    assert 0.99 <= peak_sigma <= 1.0
    for _ in range(12):
        p.grad = torch.full((4,), 0.1)
        ctrl.observe([p])
    decayed = ctrl.sigma(24)
    assert 0.1 < decayed < 0.4
    restored = ae.NoiseController(ae.Explore(1.0, 0, "grad_norm", 4.0, None))
    restored.load_state_dict(ctrl.state_dict(), torch.device("cpu"))
    assert restored.sigma(24) == pytest.approx(decayed)

    # grad_snr: stays near σ_max for aligned gradients, drops for canceling ±g, and resumes cleanly.
    snr = ae.NoiseController(ae.Explore(1.0, 0, "grad_snr", 4.0, None))
    assert snr.sigma(0) == 1.0
    for _ in range(5):
        p.grad = torch.ones(4)
        snr.observe([p])
    assert snr.sigma(5) == pytest.approx(1.0, abs=1e-5)
    for step in range(40):
        p.grad = torch.ones(4) if step % 2 == 0 else -torch.ones(4)
        snr.observe([p])
    assert 0.0 <= snr.sigma(45) < 0.2
    snr_restored = ae.NoiseController(ae.Explore(1.0, 0, "grad_snr", 4.0, None))
    snr_restored.load_state_dict(snr.state_dict(), torch.device("cpu"))
    p.grad = torch.ones(4)
    assert snr_restored.observe([p]) == pytest.approx(snr.observe([p]))
    assert snr_restored.sigma(46) == pytest.approx(snr.sigma(46))

    with pytest.raises(ValueError):
        ae.NoiseController(ae.Explore(1.0, 0, "bogus", 4.0, None))
    with pytest.raises(ValueError):
        ae.NoiseController(ae.Explore(1.0, 0, "grad_norm", 0.0, None))


def test_schedulefree_train_eval_and_resume(tmp_path):
    model = load_model(TINY)
    geometries = {name: PairGeometry(model.pair(name), model.value_count) for name in ("c512_k32", "lossless")}
    images = ImageSet(torch.randint(0, 256, (40, 1, 32, 32), dtype=torch.uint8), "cpu")
    store = ArtifactStore(tmp_path, None)
    _teachers(model, geometries, images, store)
    overrides = {
        **FAST,
        "optimizer": "schedulefree",
        "batch_size": 4,
        "warmup_steps": 4,
        "integration_steps": 2,
        "explore.noise_sigma": 1.0,
        "explore.noise_mode": "grad_snr",
        "explore.noise_half_life": 5.0,
        "validation_batch_size": 4,
        "validation_batch_count": 1,
        "gallery_count": 2,
    }
    config = load(ae.AEStageConfig, "ae", overrides)

    # Run 0..2, snapshot state_dict, then continue 2..4 alongside a resumed task.
    task = ae.AETask(model, config, geometries, images, store)
    assert task.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    torch.manual_seed(11)
    step0 = task.train_step()
    assert task.optimizer.param_groups[0]["train_mode"] is True
    assert step0["lr"] == pytest.approx(config.learning_rate / config.warmup_steps)
    # AdamWScheduleFree divides p.grad in-place during step(); observed norm must differ from post-step p.grad norm.
    post_step_norm = float(
        torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in task.encoder.parameters() if p.grad is not None]))
    )
    assert abs(step0["encoder_grad_norm"] - post_step_norm) > 1e-3

    task.train_step()
    snap_path = tmp_path / "snap.pt"
    ae.checkpoint.save(task.state_dict(), snap_path)
    assert task.optimizer.param_groups[0]["train_mode"] is False
    # Continuing from step 2 after state_dict() must switch back to train_mode and match a fresh resumed task.
    torch.manual_seed(99)
    task.train_step()
    task.train_step()
    expected_weights = {k: v.clone() for k, v in task.weights()["modules"].items()}
    expected_sigma = task.noise.sigma(task.step)
    assert task.optimizer.param_groups[0]["train_mode"] is False

    resumed = ae.AETask(model, config, geometries, images, store)
    resumed.load_state_dict(ae.checkpoint.load(snap_path))
    torch.manual_seed(99)
    resumed.train_step()
    resumed.train_step()
    actual_weights = resumed.weights()["modules"]
    assert resumed.noise.sigma(resumed.step) == pytest.approx(expected_sigma, rel=1e-6)
    for key in expected_weights:
        assert torch.allclose(expected_weights[key], actual_weights[key], atol=1e-6)


def test_registered_studies_load_cleanly():
    from lpap import experiments

    for arms in experiments.STUDIES.values():
        for arm in arms:
            cfg = load(ae.AEStageConfig, "ae", arm.overrides)
            assert cfg.optimizer in ae.OPTIMIZERS
            ctrl = ae.NoiseController(cfg.explore)
            assert ctrl.sigma(0) == (cfg.explore.noise_sigma if cfg.explore.noise_sigma > 0 else 0.0)
