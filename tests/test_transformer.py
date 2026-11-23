import torch

from lpap.transformer import TransformerBlock


def test_circular_rope_is_shift_equivariant():
    torch.manual_seed(0)
    block = TransformerBlock(16, 2)
    tokens = torch.randn(2, 12, 16)
    shifted = block(torch.roll(tokens, 5, dims=1))
    assert torch.allclose(shifted, torch.roll(block(tokens), 5, dims=1), atol=1e-5)


def test_mask_blocks_attention():
    torch.manual_seed(0)
    block = TransformerBlock(16, 2)
    count, window = 12, 3
    index = torch.arange(count)
    mask = (index[:, None] - index[None, :]) % count < window
    tokens = torch.randn(1, count, 16)
    perturbed = tokens.clone()
    perturbed[0, 6] += torch.randn(16)
    changed = (block(tokens, mask) - block(perturbed, mask)).abs().amax(-1)[0] > 1e-6
    assert torch.equal(changed, mask[:, 6])
