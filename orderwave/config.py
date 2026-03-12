from __future__ import annotations

"""Configuration models for the public ``orderwave`` API."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, cast


@dataclass(frozen=True)
class MarketConfig:
    """Compact statistical controls for ``orderwave.Market``."""

    limit_rate: float = 6.0
    market_rate: float = 2.0
    cancel_rate: float = 4.0
    fair_price_vol: float = 0.35
    mean_reversion: float = 0.08
    level_decay: float = 0.65
    size_mean: float = 1.2
    size_dispersion: float = 0.5
    min_order_qty: int = 1
    max_order_qty: int = 25
    max_spread_ticks: int = 6
    max_fair_move_ticks: int = 3

    def validate(self) -> MarketConfig:
        """Return a validated copy of the config."""

        if self.limit_rate <= 0.0:
            raise ValueError("limit_rate must be greater than 0")
        if self.market_rate <= 0.0:
            raise ValueError("market_rate must be greater than 0")
        if self.cancel_rate <= 0.0:
            raise ValueError("cancel_rate must be greater than 0")
        if not 0.0 <= self.mean_reversion <= 1.0:
            raise ValueError("mean_reversion must be between 0 and 1")
        if not 0.0 < self.level_decay < 1.0:
            raise ValueError("level_decay must be between 0 and 1")
        if self.size_dispersion <= 0.0:
            raise ValueError("size_dispersion must be greater than 0")
        if self.min_order_qty < 1:
            raise ValueError("min_order_qty must be at least 1")
        if self.max_order_qty < self.min_order_qty:
            raise ValueError("max_order_qty must be greater than or equal to min_order_qty")
        if self.max_spread_ticks < 1:
            raise ValueError("max_spread_ticks must be at least 1")
        if self.max_fair_move_ticks < 1:
            raise ValueError("max_fair_move_ticks must be at least 1")
        return self

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def coerce_config(config: MarketConfig | Mapping[str, object] | None) -> MarketConfig:
    """Coerce external config input into ``MarketConfig``."""

    if config is None:
        return MarketConfig().validate()
    if isinstance(config, MarketConfig):
        return config.validate()
    if not isinstance(config, Mapping):
        raise TypeError("config must be a MarketConfig, mapping, or None")

    valid_fields = {field.name for field in fields(MarketConfig)}
    unknown = sorted(set(config) - valid_fields)
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"unknown MarketConfig fields: {unknown_text}")

    values = cast(dict[str, Any], dict(config))
    return MarketConfig(**values).validate()


__all__ = ["MarketConfig", "coerce_config"]
