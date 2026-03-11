from __future__ import annotations

import math
from dataclasses import replace
from typing import Mapping

import numpy as np

from orderwave.config import REGIME_NAMES, MarketConfig, PresetParams, RegimeName
from orderwave.metrics import MarketFeatures
from orderwave.utils import clamp

from .types import SHOCK_NAMES, AggressorSide, EngineContext, MetaOrderState, SessionPhase, ShockState


def resolve_session_phase(session_step: int, steps_per_day: int) -> tuple[SessionPhase, float]:
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive")

    progress = clamp(session_step / max(steps_per_day, 1), 0.0, 1.0)
    if progress <= 0.2:
        return "open", progress
    if progress < 0.8:
        return "mid", progress
    return "close", progress


def seasonality_multipliers(
    session_phase: SessionPhase,
    *,
    session_progress: float,
    scale: float,
) -> dict[str, float]:
    base_profiles = {
        "open": {
            "limit": 0.86,
            "market": 1.48,
            "cancel": 1.34,
            "depth": 0.82,
            "meta": 1.18,
            "shock": 1.14,
        },
        "mid": {
            "limit": 1.12,
            "market": 0.72,
            "cancel": 0.74,
            "depth": 1.12,
            "meta": 0.84,
            "shock": 0.78,
        },
        "close": {
            "limit": 0.9,
            "market": 1.4,
            "cancel": 1.24,
            "depth": 0.88,
            "meta": 1.2,
            "shock": 1.2,
        },
    }
    weights = base_profiles[session_phase]
    scaled = {}
    for key, value in weights.items():
        scaled[key] = 1.0 + ((value - 1.0) * scale)

    edge_boost = 1.0 + (0.08 * scale * abs(session_progress - 0.5) * 2.0)
    scaled["market"] *= edge_boost
    scaled["cancel"] *= edge_boost
    return scaled


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


def advance_hidden_volatility(
    hidden_vol: float,
    *,
    features: MarketFeatures,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    seasonality = context.seasonality["market"] * context.seasonality["cancel"]
    shock_boost = 1.0
    if context.shock is not None and context.shock.name == "vol_burst":
        shock_boost += 1.5 * context.shock.intensity

    excitation_push = 0.12 * (
        context.excitation["market_buy"]
        + context.excitation["market_sell"]
        + context.excitation["cancel_bid_near"]
        + context.excitation["cancel_ask_near"]
    )
    realized = max(features.realized_vol, 0.02)
    regime_target = {
        "calm": 0.28,
        "directional": 0.44,
        "stressed": 0.72,
    }[regime]
    target = regime_target + (0.65 * realized) + (0.06 * context.spread_excess) + excitation_push
    target *= (0.9 + 0.1 * seasonality) * shock_boost
    target += 0.08 * context.recovery_pressure
    target += 0.05 * context.one_sided_pressure
    target += 0.04 * context.impact_residue

    vol_noise_scale = 0.9 + (0.08 * math.sqrt(clamp(config.fair_price_vol_scale, 0.5, 2.5)))
    next_hidden_vol = hidden_vol + (params.latent.hidden_vol_reversion * (target - hidden_vol))
    next_hidden_vol += params.latent.hidden_vol_vol * vol_noise_scale * rng.normal()
    return float(clamp(next_hidden_vol, 0.08, 3.0))


def advance_directional_anchor(
    current_anchor: float,
    *,
    features: MarketFeatures,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    params: PresetParams,
) -> float:
    profile = params.regimes[regime]
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.75)
    return_signal = _bounded_signal(features.recent_return / max(features.spread_price, 0.01), scale=1.4)
    excitation_signal = _bounded_signal(
        context.excitation["market_buy"] - context.excitation["market_sell"],
        scale=2.0,
    )
    meta_signal = _meta_signal(context.meta_orders)
    shock_signal = _shock_directional_signal(context.shock)

    persistence = clamp(0.72 + (0.55 * max(profile.fair_drift, 0.0)), 0.72, 0.96)
    if regime == "directional":
        persistence = clamp(persistence + 0.03, 0.72, 0.975)
    elif regime == "calm":
        persistence = max(0.7, persistence - 0.08)

    drift_pulse = 0.0
    if regime == "directional":
        sign_source = current_anchor
        if abs(sign_source) < 0.08:
            sign_source = meta_signal + flow_signal + excitation_signal + (0.15 * rng.normal())
        drift_sign = 1.0 if sign_source >= 0.0 else -1.0
        drift_pulse = drift_sign * (0.03 + (0.22 * max(profile.fair_drift, 0.0)))
        if abs(current_anchor) < 0.12 and rng.random() < (0.05 + (0.18 * max(profile.fair_drift, 0.0))):
            drift_pulse += (1.0 if rng.random() < 0.5 else -1.0) * (0.06 + (0.16 * max(profile.fair_drift, 0.0)))

    target = (
        (0.5 * meta_signal)
        + (0.22 * flow_signal)
        + (0.12 * return_signal)
        + (0.1 * excitation_signal)
        + (0.14 * shock_signal)
        + (0.12 * context.impact_residue * (1.0 if current_anchor >= 0.0 else -1.0 if current_anchor < 0.0 else 0.0))
        + drift_pulse
    )
    noise_scale = 0.01 + (0.035 * profile.fair_vol)
    next_anchor = (persistence * current_anchor) + ((1.0 - persistence) * target) + (noise_scale * rng.normal())
    return float(clamp(next_anchor, -1.5, 1.5))


def advance_hidden_fair_state(
    slow_component: float,
    fast_component: float,
    jump_component: float,
    *,
    features: MarketFeatures,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> tuple[float, float, float]:
    profile = params.regimes[regime]
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.7)
    depth_signal = _bounded_signal(features.depth_imbalance, scale=0.7)
    return_signal = _bounded_signal(features.recent_return / max(features.spread_price, 0.01), scale=1.2)
    meta_signal = _meta_signal(context.meta_orders)
    shock_signal = _shock_directional_signal(context.shock)
    slow_state_signal = _bounded_signal(slow_component, scale=1.0)
    fast_state_signal = _bounded_signal(fast_component, scale=0.8)
    fair_noise_scale = clamp(0.4 + (0.6 * config.fair_price_vol_scale), 0.4, 2.2)
    jump_prob_scale = clamp(0.55 + (0.45 * config.fair_price_vol_scale), 0.45, 2.1)
    jump_amp_scale = clamp(0.3 + (0.7 * config.fair_price_vol_scale), 0.4, 2.4)

    slow_target = (
        0.03 * flow_signal
        + 0.015 * depth_signal
        + 0.48 * meta_signal
        + 0.27 * shock_signal
        + (0.25 * context.directional_anchor)
        + (0.3 * max(profile.fair_drift, 0.0) * slow_state_signal)
    )
    slow_noise = params.latent.slow_fair_vol * fair_noise_scale * (0.3 + (0.35 * context.hidden_vol)) * rng.normal()
    next_slow = slow_component + params.latent.fair_mean_reversion * (slow_target - slow_component)
    next_slow += profile.fair_drift * slow_target
    next_slow += slow_noise

    fast_target = (
        0.045 * flow_signal
        + 0.02 * depth_signal
        + 0.05 * return_signal
        + 0.035 * meta_signal
        + 0.11 * shock_signal
        + (0.08 * context.directional_anchor)
        + (0.08 * slow_state_signal)
        + (0.02 * fast_state_signal)
        - 0.03 * context.imbalance_displacement
        + (0.03 * context.impact_residue * math.copysign(1.0, context.directional_anchor if context.directional_anchor != 0.0 else 1.0))
    )
    fast_noise = params.latent.fast_fair_vol * fair_noise_scale * max(0.35, math.sqrt(context.hidden_vol)) * rng.normal()
    next_fast = fast_component + params.latent.fast_fair_reversion * (fast_target - fast_component) + fast_noise

    next_jump = jump_component * 0.45
    if context.shock is not None and context.shock.name == "fair_jump":
        jump_sign = 1.0 if context.shock.side == "buy" else -1.0
        next_jump += jump_sign * context.shock.intensity * jump_amp_scale * (0.6 + 0.1 * rng.normal())
    elif rng.random() < clamp(params.latent.fair_jump_prob * jump_prob_scale, 0.0, 0.2):
        next_jump += rng.normal(0.0, params.latent.fair_jump_scale * jump_amp_scale * 0.55)

    return float(next_slow), float(next_fast), float(next_jump)


def decay_excitation(
    excitation: Mapping[str, float],
    *,
    config: MarketConfig,
    params: PresetParams,
) -> dict[str, float]:
    decay = clamp(params.latent.excitation_decay, 0.35, 0.98)
    effective_decay = clamp(decay + (0.04 * (config.excitation_scale - 1.0)), 0.25, 0.995)
    return {key: float(value) * effective_decay for key, value in excitation.items()}


def derive_burst_state(excitation: Mapping[str, float]) -> str:
    market_buy = excitation["market_buy"]
    market_sell = excitation["market_sell"]
    cancel_bid = excitation["cancel_bid_near"]
    cancel_ask = excitation["cancel_ask_near"]
    replenish = max(excitation["limit_bid_near"], excitation["limit_ask_near"])
    dominant = max(market_buy, market_sell, cancel_bid, cancel_ask, replenish)
    if dominant < 0.65:
        return "calm"
    if dominant == market_buy:
        return "market_buy_burst"
    if dominant == market_sell:
        return "market_sell_burst"
    if dominant == cancel_bid or dominant == cancel_ask:
        return "cancel_burst"
    return "replenish_burst"


def advance_meta_orders(
    meta_orders: Mapping[AggressorSide, MetaOrderState | None],
    *,
    hidden_fair_tick: float,
    features: MarketFeatures,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
    next_meta_order_id: int,
) -> tuple[dict[AggressorSide, MetaOrderState | None], int]:
    updated: dict[AggressorSide, MetaOrderState | None] = {"buy": None, "sell": None}

    for side in ("buy", "sell"):
        current = meta_orders.get(side)
        if current is not None:
            aged = replace(current, age=current.age + 1)
            max_age = max(4, int(max(aged.decay_half_life, 1) * 3.0))
            if aged.remaining_qty > 1.0 and aged.age <= max_age:
                updated[side] = aged

    fair_gap = hidden_fair_tick - features.mid_tick
    return_signal = _bounded_signal(features.recent_return / max(features.spread_price, 0.01), scale=1.5)
    active_meta_signal = _bounded_signal(_meta_signal(updated), scale=1.0)
    directional_bias = (
        1.15 * _bounded_signal(fair_gap, scale=2.5)
        + (0.55 * _bounded_signal(features.recent_flow_imbalance, scale=0.8))
        + (0.22 * return_signal)
        + (0.45 * active_meta_signal)
        + (0.6 * context.directional_anchor)
    )
    regime_scale = {"calm": 0.7, "directional": 1.15, "stressed": 0.95}[regime]
    spawn_scale = params.meta.meta_spawn_prob * config.meta_order_scale * context.seasonality["meta"] * regime_scale
    if context.shock is not None and context.shock.name == "one_sided_taker_surge":
        spawn_scale *= 1.15

    for side in ("buy", "sell"):
        if updated[side] is not None:
            continue
        scale_bias = max(config.meta_order_scale - 1.0, 0.0)
        side_sign = 1.0 if side == "buy" else -1.0
        opposite_side = "sell" if side == "buy" else "buy"
        opposite_meta = updated[opposite_side]
        directional_push = max(side_sign * directional_bias, 0.0)
        spawn_prob = spawn_scale * (0.2 + (0.9 * directional_push))
        if side_sign * active_meta_signal > 0.0:
            spawn_prob *= 1.2 + (0.15 * scale_bias)
        elif side_sign * active_meta_signal < 0.0:
            spawn_prob *= 0.45 / (1.0 + scale_bias)
        if opposite_meta is not None:
            opposition = opposite_meta.urgency * (1.0 - meta_order_progress(opposite_meta))
            if regime == "directional" and opposition > 0.08:
                continue
            if opposition > 0.2:
                spawn_prob *= 0.06 / (1.0 + (0.5 * scale_bias))
            else:
                spawn_prob *= 1.0 / (1.0 + ((4.5 + (2.0 * scale_bias)) * opposition))
        spawn_prob = clamp(spawn_prob, 0.0, 0.12)
        if rng.random() >= spawn_prob:
            continue
        qty = max(8.0, rng.lognormal(params.meta.meta_qty_log_mean, params.meta.meta_qty_log_sigma))
        qty *= 1.0 + (0.14 * scale_bias)
        urgency = float(clamp(0.55 + (0.62 * directional_push) + (0.14 * rng.normal()), 0.4, 1.4))
        urgency *= 1.0 + (0.08 * scale_bias)
        base_duration = max(6, int(rng.poisson(params.meta.meta_duration_mean) + 4))
        decay_half_life = max(6, int(base_duration * (1.0 + (0.16 * scale_bias))))
        updated[side] = MetaOrderState(
            id=next_meta_order_id,
            side=side,
            initial_qty=float(qty),
            remaining_qty=float(qty),
            urgency=urgency,
            decay_half_life=decay_half_life,
        )
        next_meta_order_id += 1

    return updated, next_meta_order_id


def advance_shock_state(
    current_shock: ShockState | None,
    *,
    features: MarketFeatures,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> ShockState | None:
    if current_shock is not None:
        if current_shock.remaining_steps <= 1:
            return None
        return replace(current_shock, remaining_steps=current_shock.remaining_steps - 1)

    regime_scale = {"calm": 0.65, "directional": 1.0, "stressed": 1.4}[regime]
    spawn_prob = params.shock.shock_spawn_prob * config.shock_scale * context.seasonality["shock"] * regime_scale
    spawn_prob *= 0.7 + (0.3 * clamp(context.hidden_vol, 0.0, 2.0))
    spawn_prob *= 1.0 + (0.12 * context.one_sided_pressure) + (0.08 * context.drought_age)
    if rng.random() >= clamp(spawn_prob, 0.0, 0.18):
        return None

    shock_name = rng.choice(SHOCK_NAMES, p=np.array([0.24, 0.26, 0.26, 0.24], dtype=float))
    duration = max(3, int(rng.poisson(params.shock.shock_duration_mean) + 2))
    intensity = float(clamp(0.55 + (0.3 * rng.random()) + (0.12 * context.hidden_vol), 0.45, 1.8))
    side = None
    if shock_name in {"fair_jump", "one_sided_taker_surge"}:
        side = "buy" if rng.random() < 0.5 else "sell"
    elif shock_name == "liquidity_drought":
        side = "bid" if rng.random() < 0.5 else "ask"

    return ShockState(name=shock_name, intensity=intensity, remaining_steps=duration, side=side)


def meta_order_progress(meta_order: MetaOrderState | None) -> float:
    if meta_order is None or meta_order.initial_qty <= 0.0:
        return 0.0
    progress = 1.0 - (meta_order.remaining_qty / meta_order.initial_qty)
    return float(clamp(progress, 0.0, 1.0))


def update_excitation_state(
    excitation: Mapping[str, float],
    *,
    event_type: str,
    side: str,
    level: int | None,
    applied_qty: float,
    fill_qty: float,
    config: MarketConfig,
) -> dict[str, float]:
    updated = dict(excitation)
    event_scale = math.log1p(max(applied_qty, fill_qty, 0.0))

    if event_type == "market":
        key = "market_buy" if side == "buy" else "market_sell"
        updated[key] += (0.9 + event_scale) * config.excitation_scale
        opposite_cancel = "cancel_ask_near" if side == "buy" else "cancel_bid_near"
        same_limit = "limit_bid_near" if side == "buy" else "limit_ask_near"
        updated[opposite_cancel] += 0.15 * config.excitation_scale
        updated[same_limit] += 0.12 * config.excitation_scale
        return updated

    if event_type == "cancel":
        near_best = level is None or level <= 1
        key = "cancel_bid_near" if side == "bid" else "cancel_ask_near"
        updated[key] += (0.55 + (0.35 if near_best else 0.1) + (0.25 * event_scale)) * config.excitation_scale
        return updated

    if level is None or level <= 1:
        key = "limit_bid_near" if side == "bid" else "limit_ask_near"
        updated[key] += (0.45 + (0.22 * event_scale)) * config.excitation_scale
    return updated


def update_resiliency_state(
    *,
    seasonality_depth: float,
    current_spread_excess: float,
    current_bid_deficit: float,
    current_ask_deficit: float,
    current_imbalance_displacement: float,
    shock: ShockState | None,
    meta_orders: Mapping[AggressorSide, MetaOrderState | None],
    spread_ticks: int,
    depth_imbalance: float,
    best_bid_qty: int,
    best_ask_qty: int,
    config: MarketConfig,
    params: PresetParams,
) -> tuple[float, float, float, float]:
    target_best_qty = max(1.0, math.exp(params.qty.limit_qty_log_mean) * seasonality_depth)
    best_bid_deficit = max(target_best_qty - best_bid_qty, 0.0) / target_best_qty
    best_ask_deficit = max(target_best_qty - best_ask_qty, 0.0) / target_best_qty
    spread_excess = max(spread_ticks - params.shape.initial_spread_ticks, 0.0)
    imbalance_target = _meta_signal(meta_orders)
    imbalance_displacement = abs(depth_imbalance - imbalance_target)

    effective_half_life = max(params.resiliency.resiliency_half_life / max(config.resiliency_scale, 1e-6), 1.0)
    decay = math.exp(-1.0 / effective_half_life)
    if shock is not None and shock.name == "liquidity_drought":
        decay = min(0.99, decay + (0.08 * config.depletion_scale))
    if any(meta_orders.values()):
        decay = min(0.99, decay + (0.05 * config.depletion_scale))

    next_spread_excess = max(spread_excess, current_spread_excess * decay)
    next_bid_deficit = max(best_bid_deficit, current_bid_deficit * decay)
    next_ask_deficit = max(best_ask_deficit, current_ask_deficit * decay)
    next_imbalance_displacement = max(imbalance_displacement, current_imbalance_displacement * decay)
    return (
        float(next_spread_excess),
        float(next_bid_deficit),
        float(next_ask_deficit),
        float(next_imbalance_displacement),
    )


def _meta_signal(meta_orders: Mapping[AggressorSide, MetaOrderState | None]) -> float:
    signal = 0.0
    buy_meta = meta_orders.get("buy")
    sell_meta = meta_orders.get("sell")
    if buy_meta is not None:
        signal += buy_meta.urgency * (1.0 - meta_order_progress(buy_meta))
    if sell_meta is not None:
        signal -= sell_meta.urgency * (1.0 - meta_order_progress(sell_meta))
    return float(clamp(signal, -2.0, 2.0))


def _shock_directional_signal(shock: ShockState | None) -> float:
    if shock is None:
        return 0.0
    if shock.name in {"fair_jump", "one_sided_taker_surge"}:
        return shock.intensity if shock.side == "buy" else -shock.intensity
    return 0.0


def _bounded_signal(value: float, *, scale: float) -> float:
    return math.tanh(value / max(scale, 1e-6))
