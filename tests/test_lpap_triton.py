import pytest
import torch

from lpap.lpap import lpap_tokens_torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize(("bucket_count", "k_max"), [(128, 16), (256, 24), (512, 32), (1024, 1), (16, 16)])
@pytest.mark.parametrize("quantized", [False, True])
def test_triton_matches_torch(bucket_count, k_max, quantized):
    from lpap.lpap_triton import lpap_tokens_triton

    generator = torch.Generator().manual_seed(bucket_count + k_max)
    tokens = torch.randn(8, bucket_count, 1024 // bucket_count, generator=generator)
    if quantized:
        tokens = (tokens * 2).round() / 2
    reference = lpap_tokens_torch(tokens, k_max=k_max)
    result = lpap_tokens_triton(tokens.cuda(), k_max=k_max)
    for field in ("amplitudes", "dibs", "probes", "remaining"):
        assert torch.equal(getattr(result, field).cpu(), getattr(reference, field)), field


def test_triton_rejects_grad():
    from lpap.lpap_triton import lpap_tokens_triton

    with pytest.raises(RuntimeError):
        lpap_tokens_triton(torch.randn(1, 8, 4, device="cuda", requires_grad=True), k_max=2)
