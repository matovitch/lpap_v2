"""Strict TOML → frozen dataclass loader (FOUNDATION §7).

Every field is required, except ``Optional`` fields, which default to ``None``.
Unknown keys are errors. Overrides use dotted keys, e.g. ``{"flow.width": 64}``.
"""

from __future__ import annotations

import dataclasses
import tomllib
import types
import typing
from dataclasses import dataclass
from importlib import resources
from typing import Any, TypeVar

from lpap.decoder import DecoderConfig
from lpap.flow import FlowConfig
from lpap.geometry import PairSpec
from lpap.surrogate import SurrogateConfig

T = TypeVar("T")


@dataclass(frozen=True)
class PriorConfig:
    sigma: float
    scale: float


@dataclass(frozen=True)
class ModelConfig:
    value_count: int
    image_side: int
    flow: FlowConfig
    prior: PriorConfig
    surrogate: SurrogateConfig
    decoder: DecoderConfig
    pairs: tuple[PairSpec, ...]

    def pair(self, name: str) -> PairSpec:
        for spec in self.pairs:
            if spec.name == name:
                return spec
        raise KeyError(name)


class ConfigError(ValueError):
    pass


def _is_optional(annotation: Any) -> bool:
    return typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in typing.get_args(
        annotation
    )


def _convert(annotation: Any, value: Any, path: str) -> Any:
    if _is_optional(annotation):
        if value is None:
            return None
        (inner,) = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return _convert(inner, value, path)
    if dataclasses.is_dataclass(annotation):
        return from_dict(annotation, value, path)
    origin = typing.get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{path}: expected a list")
        (item, ellipsis) = typing.get_args(annotation)
        assert ellipsis is Ellipsis
        return tuple(_convert(item, element, f"{path}[{index}]") for index, element in enumerate(value))
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if annotation in (int, float, str, bool):
        if type(value) is not annotation:
            raise ConfigError(f"{path}: expected {annotation.__name__}, got {type(value).__name__}")
        return value
    raise ConfigError(f"{path}: unsupported type {annotation!r}")


def from_dict(cls: type[T], data: Any, path: str = "") -> T:
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__}: expected a table")
    hints = typing.get_type_hints(cls)
    fields = {field.name for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ConfigError(f"{path or cls.__name__}: unknown keys {unknown}")
    values = {}
    for name in fields:
        where = f"{path}.{name}" if path else name
        if name not in data:
            if _is_optional(hints[name]):
                values[name] = None
                continue
            raise ConfigError(f"{where}: missing")
        values[name] = _convert(hints[name], data[name], where)
    return cls(**values)


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        table = data
        for key in parents:
            if not isinstance(table.get(key), dict):
                raise ConfigError(f"{dotted}: no table {key!r}")
            table = table[key]
        if leaf not in table:
            raise ConfigError(f"{dotted}: unknown key")
        table[leaf] = value
    return data


def load(cls: type[T], name: str, overrides: dict[str, Any] | None = None) -> T:
    """Load ``lpap/configs/<name>.toml`` into ``cls``."""
    text = resources.files("lpap.configs").joinpath(f"{name}.toml").read_text()
    return from_dict(cls, apply_overrides(tomllib.loads(text), overrides or {}))


def load_model(overrides: dict[str, Any] | None = None) -> ModelConfig:
    return load(ModelConfig, "model", overrides)


def to_dict(config: Any) -> dict[str, Any]:
    """Resolved config, stored in every checkpoint."""
    return dataclasses.asdict(config)
