import pytest
import torch

from lpap.geometry import PairGeometry, PairSpec
from lpap.surrogate import Surrogate, SurrogateConfig

CONFIG = SurrogateConfig(hidden_dim=16, layer_count=2, head_count=2, rms_eps=1e-6)


def test_shapes_and_mask():
    geometry = PairGeometry(PairSpec("t", 16, 4, 1), 64)
    surrogate = Surrogate(geometry, CONFIG)
    assert surrogate(torch.randn(3, 16, 4)).shape == (3, 16, 16)
    index = torch.arange(16)
    assert torch.equal(surrogate.attention_mask, (index[:, None] - index[None, :]) % 16 < 4)


def test_full_window_has_no_mask():
    geometry = PairGeometry(PairSpec("t", 8, 8, 1), 64)
    assert Surrogate(geometry, CONFIG).attention_mask is None


def test_input_is_scale_normalised():
    torch.manual_seed(0)
    surrogate = Surrogate(PairGeometry(PairSpec("t", 16, 4, 1), 64), CONFIG)
    tokens = torch.randn(2, 16, 4)
    assert torch.allclose(surrogate(tokens), surrogate(7.0 * tokens), atol=1e-4)


def test_lossless_pair_has_no_surrogate():
    with pytest.raises(ValueError):
        Surrogate(PairGeometry(PairSpec("lossless", 64, 1), 64), CONFIG)
