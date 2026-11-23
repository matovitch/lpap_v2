import pytest

from lpap.runtime.config import ConfigError, ModelConfig, from_dict, load_model, load_storage, to_dict


def test_model_config_loads():
    config = load_model()
    assert [pair.name for pair in config.pairs] == ["c128_k16", "c256_k24", "c512_k32", "lossless"]
    assert config.pair("lossless").permutation_seed is None
    assert config.flow.dilations == (1, 2, 4, 8, 16, 32, 64, 128)


def test_round_trip_through_dict():
    config = load_model()
    assert from_dict(ModelConfig, to_dict(config)) == config


def test_overrides():
    assert load_model({"flow.width": 64}).flow.width == 64
    with pytest.raises(ConfigError):
        load_model({"flow.typo": 1})


def test_strictness():
    data = to_dict(load_model())
    with pytest.raises(ConfigError, match="unknown"):
        from_dict(ModelConfig, {**data, "extra": 1})
    missing = dict(data)
    del missing["image_side"]
    with pytest.raises(ConfigError, match="missing"):
        from_dict(ModelConfig, missing)
    with pytest.raises(ConfigError, match="expected int"):
        from_dict(ModelConfig, {**data, "image_side": "32"})


def test_packaged_configs_load():
    assert load_storage().artifacts_repo
