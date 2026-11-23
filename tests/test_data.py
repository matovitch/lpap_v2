import torch

from lpap.data import hilbert_flatten, hilbert_order, hilbert_unflatten, load_images, split_indices, to_model_input


def test_hilbert_order_is_a_continuous_curve():
    order = hilbert_order(32)
    assert torch.equal(order.sort().values, torch.arange(1024))
    rows, cols = order // 32, order % 32
    steps = (rows.diff().abs() + cols.diff().abs()).tolist()
    assert set(steps) == {1}


def test_hilbert_round_trip():
    images = torch.rand(2, 1, 32, 32)
    assert torch.equal(hilbert_unflatten(hilbert_flatten(images)), images)


def test_split_is_deterministic_disjoint_and_complete():
    train, validation = split_indices(1000)
    again_train, again_validation = split_indices(1000)
    assert torch.equal(train, again_train) and torch.equal(validation, again_validation)
    assert validation.numel() == 50
    assert torch.equal(torch.cat((train, validation)).sort().values, torch.arange(1000))


def test_model_input_range_and_shape():
    images = torch.randint(0, 256, (3, 1, 32, 32), dtype=torch.uint8)
    values = to_model_input(images)
    assert values.shape == (3, 1, 1024)
    assert float(values.min()) >= 0.0 and float(values.max()) <= 1.0


def test_load_images_round_trip(tmp_path):
    images = torch.randint(0, 256, (4, 1, 32, 32), dtype=torch.uint8)
    path = tmp_path / "images.pt"
    torch.save({"images": images}, path)
    assert torch.equal(load_images(path), images)
