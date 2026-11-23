import torch
from torch.nn import functional as F

from lpap.losses import accuracies, flow_matching, ot_coupling, sample_bridge_time, signed_mass, weighted_ce


def test_weighted_ce_matches_manual():
    logits, labels = torch.randn(2, 5, 7), torch.randint(0, 7, (2, 5))
    weights = torch.rand(2, 5)
    manual = (weights * F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")).sum() / weights.sum()
    assert torch.allclose(weighted_ce(logits, labels, weights), manual)


def test_weighted_ce_with_zero_weights_is_zero():
    loss = weighted_ce(torch.randn(2, 5, 7), torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 5))
    assert torch.isfinite(loss) and float(loss) == 0.0


def test_accuracies():
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    labels = torch.tensor([[0, 0]])
    plain, weighted = accuracies(logits, labels, torch.tensor([[3.0, 1.0]]))
    assert float(plain) == 0.5 and float(weighted) == 0.75


def test_signed_mass():
    balanced = torch.tensor([[0.5, -0.5]])
    assert float(signed_mass(balanced, tau=0.1)) == 0.0
    assert float(signed_mass(torch.tensor([[0.5, 0.5]]), tau=0.1)) > 0.0
    assert float(signed_mass(torch.tensor([[0.01, -0.01]]), tau=0.1)) > 0.0


def test_bridge_time_covers_both_halves():
    time = sample_bridge_time(10_000, generator=torch.Generator().manual_seed(0))
    assert float(time.min()) > -1.0 and float(time.max()) < 1.0
    assert abs(float((time < 0).float().mean()) - 0.5) < 0.03
    assert float((time.abs() > 0.9).float().mean()) > 0.2


def test_flow_matching_is_zero_for_the_oracle_field():
    image, prior = torch.rand(4, 1, 8), torch.randn(4, 1, 8)
    time = torch.tensor([-0.7, -0.2, 0.3, 0.9])

    def oracle(state, t):
        return torch.where((t <= 0)[:, None, None], prior - image, image - prior)

    assert float(flow_matching(oracle, image, prior, time)) == 0.0


def test_ot_coupling_permutes_prior_to_lower_cost():
    generator = torch.Generator().manual_seed(0)
    image, prior = torch.randn(16, 1, 8, generator=generator), torch.randn(16, 1, 8, generator=generator)
    paired = ot_coupling(image, prior)
    assert sorted(paired.flatten().tolist()) == sorted(prior.flatten().tolist())
    assert (paired - image).square().sum() <= (prior - image).square().sum()
