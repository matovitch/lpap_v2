import pytest
import torch

from lpap.geometry import PairGeometry, PairSpec


def make(bucket_count=16, k_max=4, seed=7, value_count=64):
    return PairGeometry(PairSpec("test", bucket_count, k_max, seed), value_count)


def test_lane_table_is_a_row_major_permutation():
    geometry = make()
    assert geometry.lane_table.shape == (16, 4)
    assert torch.equal(geometry.lane_table.flatten().sort().values, torch.arange(64))


def test_fold_reads_the_lane_table():
    geometry = make()
    values = torch.randn(3, 64)
    tokens = geometry.fold(values)
    for b in range(16):
        for p in range(4):
            assert torch.equal(tokens[:, b, p], values[:, geometry.lane_table[b, p]])


def test_candidate_sources_and_source_indices():
    geometry = make()
    sources = geometry.candidate_sources()
    for b in range(16):
        for d in range(4):
            for p in range(4):
                assert sources[b, d, p] == geometry.lane_table[(b - d) % 16, p]
    dibs = torch.randint(0, 4, (2, 16))
    probes = torch.randint(0, 4, (2, 16))
    expected = sources[torch.arange(16), dibs, probes]
    assert torch.equal(geometry.source_indices(dibs, probes), expected)


def test_lossless_pair_is_the_identity():
    geometry = PairGeometry(PairSpec("lossless", 64, 1), 64)
    assert geometry.is_lossless
    assert torch.equal(geometry.lane_table.flatten(), torch.arange(64))


@pytest.mark.parametrize(
    ("bucket_count", "k_max", "value_count"),
    [(10, 2, 64), (16, 0, 64), (16, 17, 64), (64, 2, 64)],
)
def test_invalid_specs_are_rejected(bucket_count, k_max, value_count):
    with pytest.raises(ValueError):
        PairGeometry(PairSpec("bad", bucket_count, k_max, 1), value_count)
