from __future__ import annotations

"""Configuration models for the public `orderwave` API."""

from dataclasses import dataclass, replace
from typing import Literal, Mapping

PresetName = Literal["balanced", "trend", "volatile"]
RegimeName = Literal["calm", "directional", "stressed"]

REGIME_NAMES: tuple[RegimeName, ...] = ("calm", "directional", "stressed")
PRESET_NAMES: tuple[PresetName, ...] = ("balanced", "trend", "volatile")


@dataclass(frozen=True)
class MarketConfig:
    """High-level controls for `orderwave.Market`.

    This is the only advanced configuration object intended for external use.
    It keeps the surface compact by exposing a preset plus a small set of
    scaling knobs instead of the full internal microstructure coefficients.

    Attributes
    ----------
    preset:
        Behavior preset for the simulator. Supported values are
        ``"balanced"``, ``"trend"``, and ``"volatile"``.
    book_buffer_levels:
        Internal depth kept behind the visible ``levels`` returned by
        ``Market.get()``. Defaults to ``max(levels + 5, 10)``.
    flow_window:
        Rolling window length for signed aggressive flow features.
    vol_window:
        Rolling window length for short-horizon realized volatility features.
    limit_rate_scale:
        Multiplier applied to limit-order arrival intensity.
    market_rate_scale:
        Multiplier applied to marketable order intensity.
    cancel_rate_scale:
        Multiplier applied to cancellation intensity.
    fair_price_vol_scale:
        Multiplier applied to hidden fair-price volatility.
    regime_transition_scale:
        Multiplier applied to regime transition pressure.
    steps_per_day:
        Number of simulator steps treated as a synthetic trading session.
    seasonality_scale:
        Strength of the open/mid/close intraday seasonality profile.
    excitation_scale:
        Strength of Hawkes-like self-excitation traces.
    meta_order_scale:
        Strength of latent directional meta-order spawning and persistence.
    shock_scale:
        Strength of exogenous shock spawning and impact.
    """

    preset: PresetName = "balanced"
    book_buffer_levels: int | None = None
    flow_window: int = 20
    vol_window: int = 50
    limit_rate_scale: float = 1.0
    market_rate_scale: float = 1.0
    cancel_rate_scale: float = 1.0
    fair_price_vol_scale: float = 1.0
    regime_transition_scale: float = 1.0
    steps_per_day: int = 390
    seasonality_scale: float = 1.0
    excitation_scale: float = 1.0
    meta_order_scale: float = 1.0
    shock_scale: float = 1.0


@dataclass(frozen=True)
class RegimeProfile:
    limit_offset: float
    market_offset: float
    cancel_offset: float
    inside_offset: float
    fair_drift: float
    fair_vol: float


@dataclass(frozen=True)
class PresetParams:
    initial_spread_ticks: int
    base_shape_intercept: float
    base_shape_linear: float
    base_shape_quadratic: float
    hump_weight: float
    hump_center: float
    hump_sigma: float
    imbalance_weight: float
    fair_weight: float
    flow_weight: float
    imbalance_decay: float
    fair_decay: float
    flow_decay: float
    inside_base_bonus: float
    inside_fair_weight: float
    inside_thin_weight: float
    inside_vol_penalty: float
    stale_penalty: float
    gap_penalty: float
    limit_base_log_intensity: float
    market_base_log_intensity: float
    cancel_base_logit: float
    limit_qty_log_mean: float
    limit_qty_log_sigma: float
    market_qty_log_mean: float
    market_qty_log_sigma: float
    market_fair_weight: float
    market_flow_weight: float
    market_thin_weight: float
    market_spread_weight: float
    cancel_depth_weight: float
    cancel_vol_weight: float
    cancel_adverse_weight: float
    cancel_stale_weight: float
    fair_jump_prob: float
    fair_jump_scale: float
    fair_mean_reversion: float
    slow_fair_vol: float
    fast_fair_reversion: float
    fast_fair_vol: float
    hidden_vol_reversion: float
    hidden_vol_vol: float
    excitation_decay: float
    meta_spawn_prob: float
    meta_qty_log_mean: float
    meta_qty_log_sigma: float
    meta_duration_mean: float
    shock_spawn_prob: float
    shock_duration_mean: float
    resiliency_half_life: float
    transition_matrix: Mapping[RegimeName, Mapping[RegimeName, float]]
    regimes: Mapping[RegimeName, RegimeProfile]


_PRESETS: dict[PresetName, PresetParams] = {
    "balanced": PresetParams(
        initial_spread_ticks=1,
        base_shape_intercept=2.2,
        base_shape_linear=0.55,
        base_shape_quadratic=0.06,
        hump_weight=1.1,
        hump_center=2.5,
        hump_sigma=1.25,
        imbalance_weight=0.58,
        fair_weight=0.42,
        flow_weight=0.28,
        imbalance_decay=2.8,
        fair_decay=2.2,
        flow_decay=1.8,
        inside_base_bonus=0.9,
        inside_fair_weight=0.28,
        inside_thin_weight=0.35,
        inside_vol_penalty=1.2,
        stale_penalty=0.12,
        gap_penalty=0.05,
        limit_base_log_intensity=1.15,
        market_base_log_intensity=-1.15,
        cancel_base_logit=-2.1,
        limit_qty_log_mean=1.35,
        limit_qty_log_sigma=0.55,
        market_qty_log_mean=1.0,
        market_qty_log_sigma=0.45,
        market_fair_weight=0.42,
        market_flow_weight=0.32,
        market_thin_weight=0.7,
        market_spread_weight=0.45,
        cancel_depth_weight=0.16,
        cancel_vol_weight=1.6,
        cancel_adverse_weight=1.1,
        cancel_stale_weight=0.12,
        fair_jump_prob=0.015,
        fair_jump_scale=1.2,
        fair_mean_reversion=0.14,
        slow_fair_vol=0.06,
        fast_fair_reversion=0.32,
        fast_fair_vol=0.22,
        hidden_vol_reversion=0.14,
        hidden_vol_vol=0.08,
        excitation_decay=0.74,
        meta_spawn_prob=0.018,
        meta_qty_log_mean=2.4,
        meta_qty_log_sigma=0.45,
        meta_duration_mean=26.0,
        shock_spawn_prob=0.008,
        shock_duration_mean=12.0,
        resiliency_half_life=7.0,
        transition_matrix={
            "calm": {"calm": 0.9, "directional": 0.06, "stressed": 0.04},
            "directional": {"calm": 0.14, "directional": 0.74, "stressed": 0.12},
            "stressed": {"calm": 0.18, "directional": 0.24, "stressed": 0.58},
        },
        regimes={
            "calm": RegimeProfile(0.12, -0.18, -0.18, 0.0, 0.0, 0.34),
            "directional": RegimeProfile(0.04, 0.1, 0.05, 0.08, 0.05, 0.42),
            "stressed": RegimeProfile(-0.08, 0.28, 0.24, -0.1, 0.02, 0.62),
        },
    ),
    "trend": PresetParams(
        initial_spread_ticks=1,
        base_shape_intercept=2.05,
        base_shape_linear=0.5,
        base_shape_quadratic=0.05,
        hump_weight=0.95,
        hump_center=2.0,
        hump_sigma=1.3,
        imbalance_weight=0.72,
        fair_weight=0.62,
        flow_weight=0.42,
        imbalance_decay=2.5,
        fair_decay=2.0,
        flow_decay=1.6,
        inside_base_bonus=1.0,
        inside_fair_weight=0.4,
        inside_thin_weight=0.35,
        inside_vol_penalty=1.0,
        stale_penalty=0.1,
        gap_penalty=0.04,
        limit_base_log_intensity=1.1,
        market_base_log_intensity=-1.0,
        cancel_base_logit=-2.0,
        limit_qty_log_mean=1.3,
        limit_qty_log_sigma=0.5,
        market_qty_log_mean=1.05,
        market_qty_log_sigma=0.45,
        market_fair_weight=0.62,
        market_flow_weight=0.46,
        market_thin_weight=0.75,
        market_spread_weight=0.4,
        cancel_depth_weight=0.14,
        cancel_vol_weight=1.5,
        cancel_adverse_weight=1.2,
        cancel_stale_weight=0.1,
        fair_jump_prob=0.018,
        fair_jump_scale=1.6,
        fair_mean_reversion=0.12,
        slow_fair_vol=0.08,
        fast_fair_reversion=0.28,
        fast_fair_vol=0.28,
        hidden_vol_reversion=0.12,
        hidden_vol_vol=0.09,
        excitation_decay=0.78,
        meta_spawn_prob=0.026,
        meta_qty_log_mean=2.55,
        meta_qty_log_sigma=0.48,
        meta_duration_mean=34.0,
        shock_spawn_prob=0.01,
        shock_duration_mean=14.0,
        resiliency_half_life=9.0,
        transition_matrix={
            "calm": {"calm": 0.9, "directional": 0.07, "stressed": 0.03},
            "directional": {"calm": 0.08, "directional": 0.84, "stressed": 0.08},
            "stressed": {"calm": 0.08, "directional": 0.22, "stressed": 0.7},
        },
        regimes={
            "calm": RegimeProfile(0.1, -0.2, -0.15, 0.05, 0.02, 0.3),
            "directional": RegimeProfile(0.05, 0.25, 0.12, 0.15, 0.12, 0.55),
            "stressed": RegimeProfile(-0.1, 0.5, 0.4, -0.1, 0.05, 0.95),
        },
    ),
    "volatile": PresetParams(
        initial_spread_ticks=2,
        base_shape_intercept=1.95,
        base_shape_linear=0.52,
        base_shape_quadratic=0.07,
        hump_weight=1.25,
        hump_center=3.0,
        hump_sigma=1.4,
        imbalance_weight=0.54,
        fair_weight=0.52,
        flow_weight=0.3,
        imbalance_decay=2.6,
        fair_decay=2.1,
        flow_decay=1.7,
        inside_base_bonus=0.8,
        inside_fair_weight=0.24,
        inside_thin_weight=0.45,
        inside_vol_penalty=1.35,
        stale_penalty=0.14,
        gap_penalty=0.06,
        limit_base_log_intensity=1.0,
        market_base_log_intensity=-0.85,
        cancel_base_logit=-1.8,
        limit_qty_log_mean=1.45,
        limit_qty_log_sigma=0.6,
        market_qty_log_mean=1.15,
        market_qty_log_sigma=0.55,
        market_fair_weight=0.5,
        market_flow_weight=0.34,
        market_thin_weight=0.85,
        market_spread_weight=0.3,
        cancel_depth_weight=0.2,
        cancel_vol_weight=1.9,
        cancel_adverse_weight=1.15,
        cancel_stale_weight=0.15,
        fair_jump_prob=0.025,
        fair_jump_scale=2.2,
        fair_mean_reversion=0.11,
        slow_fair_vol=0.09,
        fast_fair_reversion=0.24,
        fast_fair_vol=0.36,
        hidden_vol_reversion=0.1,
        hidden_vol_vol=0.12,
        excitation_decay=0.82,
        meta_spawn_prob=0.022,
        meta_qty_log_mean=2.5,
        meta_qty_log_sigma=0.5,
        meta_duration_mean=24.0,
        shock_spawn_prob=0.015,
        shock_duration_mean=16.0,
        resiliency_half_life=11.0,
        transition_matrix={
            "calm": {"calm": 0.88, "directional": 0.06, "stressed": 0.06},
            "directional": {"calm": 0.09, "directional": 0.78, "stressed": 0.13},
            "stressed": {"calm": 0.08, "directional": 0.18, "stressed": 0.74},
        },
        regimes={
            "calm": RegimeProfile(0.05, -0.05, 0.0, -0.05, 0.0, 0.4),
            "directional": RegimeProfile(0.0, 0.2, 0.2, 0.05, 0.08, 0.7),
            "stressed": RegimeProfile(-0.2, 0.6, 0.55, -0.2, 0.05, 1.2),
        },
    ),
}


def coerce_config(config: MarketConfig | Mapping[str, object] | None, levels: int) -> MarketConfig:
    """Normalize user-provided config into a validated ``MarketConfig``."""

    if config is None:
        market_config = MarketConfig()
    elif isinstance(config, MarketConfig):
        market_config = config
    elif isinstance(config, Mapping):
        market_config = MarketConfig(**config)
    else:
        raise TypeError("config must be None, MarketConfig, or a mapping")

    if market_config.preset not in PRESET_NAMES:
        raise ValueError(f"unsupported preset: {market_config.preset}")
    if levels <= 0:
        raise ValueError("levels must be positive")

    resolved_buffer = market_config.book_buffer_levels
    if resolved_buffer is None:
        resolved_buffer = max(levels + 5, 10)

    if resolved_buffer < levels:
        raise ValueError("book_buffer_levels must be greater than or equal to levels")
    if market_config.flow_window <= 0 or market_config.vol_window <= 0:
        raise ValueError("flow_window and vol_window must be positive")
    if market_config.steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive")

    for field_name in (
        "limit_rate_scale",
        "market_rate_scale",
        "cancel_rate_scale",
        "fair_price_vol_scale",
        "regime_transition_scale",
        "seasonality_scale",
        "excitation_scale",
        "meta_order_scale",
        "shock_scale",
    ):
        if getattr(market_config, field_name) <= 0.0:
            raise ValueError(f"{field_name} must be positive")

    return replace(market_config, book_buffer_levels=int(resolved_buffer))


def preset_params(preset: PresetName) -> PresetParams:
    """Return the internal parameter bundle for a named preset."""

    return _PRESETS[preset]
