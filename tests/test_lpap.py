import pytest
import torch

from lpap.geometry import PairGeometry, PairSpec
from lpap.lpap import exact_code, lpap_tokens_torch


@pytest.mark.parametrize(("bucket_count", "k_max"), [(16, 1), (16, 4), (8, 8), (32, 3)])
def test_code_invariants(bucket_count, k_max):
    geometry = PairGeometry(PairSpec("t", bucket_count, k_max, 3), 64)
    values = torch.randn(5, 64, generator=torch.Generator().manual_seed(0))
    code = exact_code(values, geometry, use_triton=False)
    result = lpap_tokens_torch(geometry.fold(values), k_max=k_max)

    filled = code.amplitudes != 0
    assert bool((code.dibs < k_max).all())
    assert torch.equal(code.amplitudes, values.gather(1, code.sources))
    assert torch.equal(code.local_classes, code.dibs * geometry.probe_count + code.probes)
    for row in range(values.shape[0]):
        used = code.sources[row][filled[row]]
        assert used.unique().numel() == used.numel()
    kept = torch.cat((code.amplitudes.flatten(), result.remaining.flatten()))
    kept = kept[kept != 0].sort().values
    assert torch.equal(kept, values.flatten().sort().values)


def test_single_roll_takes_the_lane_maximum():
    geometry = PairGeometry(PairSpec("t", 16, 1, 3), 64)
    values = torch.randn(4, 64)
    code = exact_code(values, geometry, use_triton=False)
    tokens = geometry.fold(values)
    best = tokens.abs().argmax(-1)
    assert torch.equal(code.probes, best)
    assert bool((code.dibs == 0).all())


def test_ties_pick_the_lowest_probe():
    tokens = torch.ones(1, 4, 4)
    result = lpap_tokens_torch(tokens, k_max=1)
    assert bool((result.probes == 0).all())


def test_lossless_code_is_the_energy():
    geometry = PairGeometry(PairSpec("lossless", 64, 1), 64)
    values = torch.randn(3, 64)
    code = exact_code(values, geometry, use_triton=False)
    assert torch.equal(code.amplitudes, values)
    assert torch.equal(code.sources, torch.arange(64).expand(3, -1))


def test_batch_dimensions_are_flattened_consistently():
    tokens = torch.randn(2, 3, 8, 4)
    batched = lpap_tokens_torch(tokens, k_max=3)
    single = lpap_tokens_torch(tokens[1, 2], k_max=3)
    assert torch.equal(batched.amplitudes[1, 2], single.amplitudes)
    assert torch.equal(batched.dibs[1, 2], single.dibs)
