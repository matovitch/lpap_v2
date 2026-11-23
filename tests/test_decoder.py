import pytest
import torch

from lpap.decoder import Decoder, DecoderConfig
from lpap.geometry import PairGeometry, PairSpec

CONFIG = DecoderConfig(hidden_dim=16, layer_count=2, head_count=2)


def test_shapes_and_source_lanes():
    geometry = PairGeometry(PairSpec("t", 16, 4, 1), 64)
    decoder = Decoder(geometry, CONFIG)
    weights = torch.nn.functional.one_hot(torch.randint(0, 4, (3, 16)), 4).float()
    assert decoder(torch.randn(3, 16), weights).shape == (3, 16, 4)
    assert torch.equal(decoder.source_lanes, geometry.source_lanes())


def test_dib_weights_select_the_source_lane_embedding():
    torch.manual_seed(0)
    geometry = PairGeometry(PairSpec("t", 16, 4, 1), 64)
    decoder = Decoder(geometry, CONFIG)
    weights = torch.zeros(1, 16, 4, requires_grad=True)
    decoder(torch.randn(1, 16), weights).square().sum().backward()
    assert weights.grad is not None and bool((weights.grad != 0).all())


def test_lossless_pair_has_no_decoder():
    with pytest.raises(ValueError):
        Decoder(PairGeometry(PairSpec("lossless", 64, 1), 64), CONFIG)
