"""Every §3 behaviour of FOUNDATION.md."""

import pytest
import torch
from torch.nn import functional as F

from lpap.codec import decoded_energy, frontend, surrogate_through, visible_candidates
from lpap.decoder import Decoder, DecoderConfig
from lpap.geometry import PairGeometry, PairSpec
from lpap.losses import probe_ce, surrogate_ce
from lpap.lpap import exact_code
from lpap.surrogate import Surrogate, SurrogateConfig

GEOMETRY = PairGeometry(PairSpec("t", 16, 4, 7), 64)
BATCH = 3


@pytest.fixture
def energy():
    return torch.randn(BATCH, 64, generator=torch.Generator().manual_seed(0)).requires_grad_()


def make_surrogate():
    torch.manual_seed(1)
    return Surrogate(GEOMETRY, SurrogateConfig(hidden_dim=16, layer_count=2, head_count=2, rms_eps=1e-6))


def make_decoder():
    torch.manual_seed(2)
    return Decoder(GEOMETRY, DecoderConfig(hidden_dim=16, layer_count=2, head_count=2))


def test_visible_candidates_layout(energy):
    tokens = GEOMETRY.fold(energy.detach())
    candidates = visible_candidates(tokens, GEOMETRY.k_max)
    sources = GEOMETRY.candidate_sources().flatten(-2)
    assert torch.equal(candidates, energy.detach()[:, sources])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_logits", [False, True])
def test_frontend_forward_is_the_exact_code(energy, dtype, with_logits):
    values = energy.detach().to(dtype)
    code = exact_code(values, GEOMETRY, use_triton=False)
    logits = torch.randn(BATCH, 16, GEOMETRY.class_count, dtype=dtype) if with_logits else None
    inputs = frontend(GEOMETRY.fold(values), code, GEOMETRY, logits)
    assert torch.equal(inputs.amplitudes, code.amplitudes)
    assert torch.equal(inputs.dib_weights, F.one_hot(code.dibs, GEOMETRY.k_max).to(dtype))
    assert torch.equal(inputs.dibs, code.dibs)


def test_direct_amplitude_gradient_only_at_the_winner(energy):
    code = exact_code(energy, GEOMETRY, use_triton=False)
    inputs = frontend(GEOMETRY.fold(energy), code, GEOMETRY)
    assert not inputs.dib_weights.requires_grad
    upstream = torch.randn(BATCH, 16)
    (inputs.amplitudes * upstream).sum().backward()
    expected = torch.zeros(BATCH, 64).scatter_add(1, code.sources, upstream)
    assert torch.equal(energy.grad, expected)


def test_logit_gradient_is_the_soft_selection_gradient(energy):
    code = exact_code(energy, GEOMETRY, use_triton=False)
    tokens = GEOMETRY.fold(energy.detach())
    logits = torch.randn(BATCH, 16, GEOMETRY.class_count, requires_grad=True)
    upstream_a, upstream_w = torch.randn(BATCH, 16), torch.randn(BATCH, 16, GEOMETRY.k_max)

    inputs = frontend(tokens, code, GEOMETRY, logits)
    ((inputs.amplitudes * upstream_a).sum() + (inputs.dib_weights * upstream_w).sum()).backward()
    straight_through = logits.grad.clone()

    logits.grad = None
    soft = torch.softmax(logits, -1)
    soft_a = (soft * visible_candidates(tokens, GEOMETRY.k_max)).sum(-1)
    soft_w = soft.unflatten(-1, (GEOMETRY.k_max, GEOMETRY.probe_count)).sum(-1)
    ((soft_a * upstream_a).sum() + (soft_w * upstream_w).sum()).backward()
    assert torch.allclose(straight_through, logits.grad, atol=1e-6)


def test_frontend_surrogate_call_reaches_tokens_not_params(energy):
    surrogate = make_surrogate()
    logits = surrogate_through(surrogate, GEOMETRY.fold(energy))
    logits.square().sum().backward()
    assert energy.grad is not None and bool(energy.grad.abs().sum() > 0)
    assert all(param.grad is None for param in surrogate.parameters())


def test_decoded_energy_forward_is_one_spike_per_bucket(energy):
    code = exact_code(energy, GEOMETRY, use_triton=False)
    inputs = frontend(GEOMETRY.fold(energy), code, GEOMETRY)
    probe_logits = torch.randn(BATCH, 16, GEOMETRY.probe_count)
    decoded = decoded_energy(inputs, probe_logits, GEOMETRY)
    chosen = GEOMETRY.source_indices(code.dibs, probe_logits.argmax(-1))
    expected = torch.zeros(BATCH, 64).scatter_add(1, chosen, code.amplitudes)
    assert torch.equal(decoded.detach(), expected)

    oracle = F.one_hot(code.probes, GEOMETRY.probe_count).float()
    decoded = decoded_energy(inputs, oracle, GEOMETRY).detach()
    assert torch.equal(decoded.gather(1, code.sources), code.amplitudes)


def test_split_probe_st_gradients(energy):
    code = exact_code(energy, GEOMETRY, use_triton=False)
    amplitudes = code.amplitudes.clone().requires_grad_()
    dib_weights = F.one_hot(code.dibs, GEOMETRY.k_max).float().requires_grad_()
    probe_logits = torch.randn(BATCH, 16, GEOMETRY.probe_count, requires_grad=True)
    upstream = torch.randn(BATCH, 64)

    from lpap.codec import DecoderInputs

    decoded = decoded_energy(DecoderInputs(amplitudes, dib_weights, code.dibs), probe_logits, GEOMETRY)
    (decoded * upstream).sum().backward()

    chosen = GEOMETRY.source_indices(code.dibs, probe_logits.argmax(-1))
    assert torch.equal(amplitudes.grad, upstream.gather(1, chosen))

    reference_logits = probe_logits.detach().clone().requires_grad_()
    reference_weights = dib_weights.detach().clone().requires_grad_()
    site_upstream = upstream[:, GEOMETRY.candidate_sources()]
    soft = (
        reference_weights[..., None]
        * torch.softmax(reference_logits, -1)[:, :, None, :]
        * code.amplitudes[:, :, None, None]
        * site_upstream
    ).sum()
    soft.backward()
    assert torch.allclose(probe_logits.grad, reference_logits.grad, atol=1e-6)
    assert torch.allclose(dib_weights.grad, reference_weights.grad, atol=1e-6)
    assert bool(dib_weights.grad.abs().sum() > 0)


def test_full_pair_routing(energy):
    """CEs never reach the energy; reconstruction never reaches the surrogate params."""
    surrogate, decoder = make_surrogate(), make_decoder()
    upstream = torch.randn(BATCH, 64)

    def losses():
        tokens = GEOMETRY.fold(energy)
        code = exact_code(energy, GEOMETRY, use_triton=False)
        inputs = frontend(tokens, code, GEOMETRY, surrogate_through(surrogate, tokens))
        reconstruction = (
            decoded_energy(inputs, decoder(inputs.amplitudes, inputs.dib_weights), GEOMETRY) * upstream
        ).sum()
        ce_surrogate = surrogate_ce(surrogate(tokens.detach()), code)
        detached = inputs.detach()
        ce_probe = probe_ce(decoder(detached.amplitudes, detached.dib_weights), code)
        return reconstruction, ce_surrogate + ce_probe

    def grads(loss):
        modules = list(surrogate.parameters()) + list(decoder.parameters())
        for tensor in [energy, *modules]:
            tensor.grad = None
        loss.backward()
        energy_grad = energy.grad.clone() if energy.grad is not None else torch.zeros_like(energy)
        surrogate_grads = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in surrogate.parameters()
        ]
        return energy_grad, surrogate_grads

    reconstruction, ce = losses()
    energy_recon, surrogate_recon = grads(reconstruction)
    assert all(bool((g == 0).all()) for g in surrogate_recon)

    reconstruction, ce = losses()
    energy_ce, surrogate_ce_grads = grads(ce)
    assert bool((energy_ce == 0).all())
    assert any(bool((g != 0).any()) for g in surrogate_ce_grads)

    reconstruction, ce = losses()
    energy_total, _ = grads(reconstruction + ce)
    assert torch.allclose(energy_total, energy_recon, atol=1e-6)
    assert bool(energy_recon.abs().sum() > 0)
