from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from orderwave.book import OrderBook
from orderwave.config import MarketConfig, PresetParams, REGIME_NAMES, RegimeName
from orderwave.metrics import MarketFeatures
from orderwave.utils import clamp, clipped_exp, coerce_quantity, sigmoid, stable_softmax

ModelSide = Literal["bid", "ask"]
AggressorSide = Literal["buy", "sell"]


def sample_next_regime(
    current_regime: RegimeName,
    *,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> RegimeName:
    base_row = params.transition_matrix[current_regime]
    scaled_row = {}
    off_diagonal_total = 0.0
    for regime in REGIME_NAMES:
        if regime == current_regime:
            continue
        scaled_prob = base_row[regime] * config.regime_transition_scale
        scaled_row[regime] = max(0.0, scaled_prob)
        off_diagonal_total += scaled_row[regime]

    stay_prob = max(0.05, 1.0 - off_diagonal_total)
    probabilities = np.array(
        [stay_prob if regime == current_regime else scaled_row[regime] for regime in REGIME_NAMES],
        dtype=float,
    )
    probabilities /= probabilities.sum()
    return rng.choice(REGIME_NAMES, p=probabilities)


def advance_hidden_fair_tick(
    hidden_fair_tick: float,
    *,
    features: MarketFeatures,
    regime: RegimeName,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    profile = params.regimes[regime]
    flow_signal = _bounded_unit_signal(features.recent_flow_imbalance, scale=0.7)
    depth_signal = _bounded_unit_signal(features.depth_imbalance, scale=0.7)
    fair_gap = hidden_fair_tick - features.mid_tick
    fair_gap_signal = _bounded_unit_signal(fair_gap, scale=2.75)
    flow_bias = 0.35 * flow_signal + 0.15 * depth_signal
    target_tick = features.mid_tick + 0.7 * flow_bias
    fair_gap = hidden_fair_tick - features.mid_tick
    drift_push = 0.45 * flow_signal + 0.15 * depth_signal + 0.1 * fair_gap_signal

    noise = profile.fair_vol * config.fair_price_vol_scale * rng.normal()
    jump = 0.0
    if rng.random() < params.fair_jump_prob:
        jump = rng.normal(0.0, params.fair_jump_scale)

    next_fair_tick = hidden_fair_tick
    next_fair_tick += profile.fair_drift * drift_push
    next_fair_tick += noise + jump
    next_fair_tick -= params.fair_mean_reversion * (hidden_fair_tick - target_tick)
    return float(next_fair_tick)


def score_limit_levels(
    side: ModelSide,
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    config: MarketConfig,
    params: PresetParams,
    allow_inside: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    side_sign = 1.0 if side == "bid" else -1.0
    fair_gap_ticks = hidden_fair_tick - features.mid_tick
    fair_signal = _bounded_unit_signal(fair_gap_ticks, scale=2.5)
    imbalance_signal = _bounded_unit_signal(features.depth_imbalance, scale=0.75)
    flow_signal = _bounded_unit_signal(features.recent_flow_imbalance, scale=0.75)
    vol_scale = clamp(features.rolling_volatility / max(features.spread_price, 0.01), 0.0, 2.5)

    levels = np.arange(-1, config.book_buffer_levels + 1, dtype=int)
    scores = np.full(levels.shape, -np.inf, dtype=float)

    for idx, level in enumerate(levels):
        if level == -1 and (not allow_inside or book.spread_ticks <= 1):
            continue

        target_tick = book.resolve_limit_tick(side, int(level))
        if target_tick is None:
            continue

        distance = 0.0 if level < 0 else float(level)
        hump = -((distance - params.hump_center) ** 2) / (2.0 * (params.hump_sigma**2))
        base_score = (
            params.base_shape_intercept
            - params.base_shape_linear * distance
            - params.base_shape_quadratic * (distance**2)
            + params.hump_weight * math.exp(hump)
        )
        decay_imb = math.exp(-distance / params.imbalance_decay)
        decay_fair = math.exp(-distance / params.fair_decay)
        decay_flow = math.exp(-distance / params.flow_decay)

        score = base_score
        score += side_sign * params.imbalance_weight * imbalance_signal * decay_imb
        score += side_sign * params.fair_weight * fair_signal * decay_fair
        score += side_sign * params.flow_weight * flow_signal * decay_flow
        score += params.regimes[regime].limit_offset
        score -= params.stale_penalty * min(book.level_age(side, target_tick) / 10.0, 1.5)
        score -= params.gap_penalty * max(distance - 4.0, 0.0)

        if level == -1:
            thinness = features.thin_bid_best if side == "bid" else features.thin_ask_best
            directional_fair = max(side_sign * fair_signal, 0.0)
            score += params.inside_base_bonus
            score += params.inside_fair_weight * directional_fair
            score += params.inside_thin_weight * thinness
            score -= params.inside_vol_penalty * vol_scale * 0.2
            score += params.regimes[regime].inside_offset

        scores[idx] = score

    return levels, stable_softmax(scores)


def compute_limit_lambda(
    side: ModelSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    side_sign = 1.0 if side == "bid" else -1.0
    fair_gap_ticks = hidden_fair_tick - features.mid_tick
    fair_signal = _bounded_unit_signal(fair_gap_ticks, scale=2.75)
    imbalance_signal = _bounded_unit_signal(features.depth_imbalance, scale=0.8)
    flow_signal = _bounded_unit_signal(features.recent_flow_imbalance, scale=0.8)
    vol_scale = clamp(features.rolling_volatility / max(features.spread_price, 0.01), 0.0, 2.5)
    thinness = features.thin_bid_best if side == "bid" else features.thin_ask_best

    log_lambda = params.limit_base_log_intensity + params.regimes[regime].limit_offset
    log_lambda += 0.28 * side_sign * imbalance_signal
    log_lambda += 0.18 * side_sign * fair_signal
    log_lambda += 0.12 * side_sign * flow_signal
    log_lambda -= 0.1 * vol_scale
    log_lambda += 0.12 * thinness
    return max(0.05, clipped_exp(log_lambda, low=-2.0, high=3.5) * config.limit_rate_scale)


def compute_market_lambda(
    aggressor_side: AggressorSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    fair_gap_ticks = hidden_fair_tick - features.mid_tick
    profile = params.regimes[regime]
    fair_signal = _bounded_unit_signal(fair_gap_ticks, scale=2.5)
    flow_signal = _bounded_unit_signal(features.recent_flow_imbalance, scale=0.75)
    if aggressor_side == "buy":
        fair_push = max(fair_signal, 0.0)
        flow_push = max(flow_signal, 0.0)
        thin_opposite = features.thin_ask_best
    else:
        fair_push = max(-fair_signal, 0.0)
        flow_push = max(-flow_signal, 0.0)
        thin_opposite = features.thin_bid_best

    log_lambda = params.market_base_log_intensity + profile.market_offset
    log_lambda += params.market_fair_weight * fair_push
    log_lambda += params.market_flow_weight * flow_push
    log_lambda += params.market_thin_weight * thin_opposite
    log_lambda -= params.market_spread_weight * max(features.spread_ticks - 1, 0)
    return max(0.01, clipped_exp(log_lambda, low=-4.0, high=2.5) * config.market_rate_scale)


def compute_cancel_probability(
    side: ModelSide,
    *,
    depth_index: int,
    staleness: int,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    fair_gap_ticks = hidden_fair_tick - features.mid_tick
    fair_signal = _bounded_unit_signal(fair_gap_ticks, scale=2.5)
    flow_signal = _bounded_unit_signal(features.recent_flow_imbalance, scale=0.8)
    if side == "ask":
        adverse_signal = max(fair_signal, 0.0) + 0.35 * max(flow_signal, 0.0)
    else:
        adverse_signal = max(-fair_signal, 0.0) + 0.35 * max(-flow_signal, 0.0)

    vol_scale = clamp(features.rolling_volatility / max(features.spread_price, 0.01), 0.0, 2.5)
    logit = params.cancel_base_logit + params.regimes[regime].cancel_offset
    logit += params.cancel_depth_weight * depth_index
    logit += params.cancel_vol_weight * vol_scale
    logit += params.cancel_adverse_weight * adverse_signal
    logit += params.cancel_stale_weight * min(staleness / 10.0, 2.0)
    return float(clamp(sigmoid(logit) * config.cancel_rate_scale, 0.0, 0.95))


def sample_limit_events(
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for side in ("bid", "ask"):
        levels, probabilities = score_limit_levels(
            side,
            book=book,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            config=config,
            params=params,
        )
        n_orders = int(
            rng.poisson(
                compute_limit_lambda(
                    side,
                    features=features,
                    hidden_fair_tick=hidden_fair_tick,
                    regime=regime,
                    config=config,
                    params=params,
                )
            )
        )
        if n_orders <= 0:
            continue

        chosen_levels = rng.choice(levels, size=n_orders, replace=True, p=probabilities)
        for level in chosen_levels:
            events.append(
                {
                    "type": "limit",
                    "side": side,
                    "level": int(level),
                    "qty": sample_limit_qty(rng=rng, params=params),
                }
            )
    return events


def sample_market_events(
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for aggressor_side in ("buy", "sell"):
        intensity = compute_market_lambda(
            aggressor_side,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            config=config,
            params=params,
        )
        n_orders = int(rng.poisson(intensity))
        for _ in range(max(0, n_orders)):
            events.append(
                {
                    "type": "market",
                    "side": aggressor_side,
                    "qty": sample_market_qty(
                        aggressor_side,
                        features=features,
                        hidden_fair_tick=hidden_fair_tick,
                        rng=rng,
                        params=params,
                    ),
                }
            )
    return events


def sample_cancel_events(
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for side in ("bid", "ask"):
        for depth_index, (tick, qty) in enumerate(book.all_levels(side)):
            probability = compute_cancel_probability(
                side,
                depth_index=depth_index,
                staleness=book.level_age(side, tick),
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                regime=regime,
                config=config,
                params=params,
            )
            total_canceled = int(rng.binomial(int(qty), probability))
            if total_canceled <= 0:
                continue
            for chunk_qty in _split_cancel_quantity(total_canceled, rng=rng, params=params):
                events.append(
                    {
                        "type": "cancel",
                        "side": side,
                        "level": depth_index,
                        "tick": tick,
                        "qty": chunk_qty,
                    }
                )
    return events


def sample_limit_qty(*, rng: np.random.Generator, params: PresetParams) -> int:
    base_qty = rng.lognormal(mean=params.limit_qty_log_mean, sigma=params.limit_qty_log_sigma)
    return coerce_quantity(base_qty)


def sample_market_qty(
    aggressor_side: AggressorSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    rng: np.random.Generator,
    params: PresetParams,
) -> int:
    fair_gap_ticks = hidden_fair_tick - features.mid_tick
    directional_push = max(fair_gap_ticks, 0.0) if aggressor_side == "buy" else max(-fair_gap_ticks, 0.0)
    size_scale = 1.0 + 0.15 * directional_push
    base_qty = rng.lognormal(mean=params.market_qty_log_mean, sigma=params.market_qty_log_sigma)
    return coerce_quantity(base_qty * size_scale)


def _bounded_unit_signal(value: float, *, scale: float) -> float:
    safe_scale = max(scale, 1e-6)
    return math.tanh(value / safe_scale)


def _split_cancel_quantity(total_qty: int, *, rng: np.random.Generator, params: PresetParams) -> list[int]:
    remaining = int(total_qty)
    chunks: list[int] = []
    if remaining <= 0:
        return chunks

    mean_base = max(1.0, math.exp(params.limit_qty_log_mean - 0.55))
    sigma = max(0.25, params.limit_qty_log_sigma * 0.6)
    while remaining > 0:
        chunk = coerce_quantity(rng.lognormal(mean=math.log(mean_base), sigma=sigma))
        chunk = min(chunk, remaining)
        chunks.append(chunk)
        remaining -= chunk
    return chunks
