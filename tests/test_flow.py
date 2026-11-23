import torch

from lpap.flow import FlowConfig, FlowField, encode, integrate, render, sample_prior

CONFIG = FlowConfig(width=16, time_dim=8, dilation_cycles=1, dilations=(1, 2), kernel_size=3)


def test_zero_initialised_field_integrates_to_identity():
    field = FlowField(CONFIG)
    image = torch.rand(2, 1, 64)
    assert torch.equal(encode(field, image, steps=4), image.squeeze(1))
    assert torch.equal(render(field, image.squeeze(1), steps=4), image)


def test_integration_stays_fp32_under_autocast():
    field = FlowField(CONFIG)
    torch.nn.init.normal_(field.output.weight)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = integrate(field, torch.rand(2, 1, 64), t0=-1.0, t1=0.0, steps=2)
    assert out.dtype == torch.float32


def test_prior_is_signed_log_normal():
    generator = torch.Generator().manual_seed(0)
    draws = sample_prior((200_000,), sigma=1.0, scale=0.2, generator=generator)
    assert abs(float((draws > 0).float().mean()) - 0.5) < 0.01
    log_magnitude = (draws.abs() / 0.2).log()
    assert abs(float(log_magnitude.mean())) < 0.01
    assert abs(float(log_magnitude.std()) - 1.0) < 0.01
