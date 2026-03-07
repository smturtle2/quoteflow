from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from orderwave.book import OrderBook
from orderwave.config import MarketConfig, PresetParams, REGIME_NAMES, RegimeName
from orderwave.metrics import MarketFeatures
from orderwave.utils import clamp, clipped_exp, coerce_quantity, stable_softmax

ModelSide = Literal["bid", "ask"]
AggressorSide = Literal["buy", "sell"]
SessionPhase = Literal["open", "mid", "close"]
ParticipantType = Literal["passive_lp", "inventory_mm", "noise_taker", "informed_meta"]
ShockName = Literal["fair_jump", "vol_burst", "liquidity_drought", "one_sided_taker_surge"]


@dataclass(frozen=True)
class MetaOrderState:
    id: int
    side: AggressorSide
    initial_qty: float
    remaining_qty: float
    urgency: float
    decay_half_life: int
    age: int = 0


@dataclass(frozen=True)
class ShockState:
    name: ShockName
    intensity: float
    remaining_steps: int
    side: str | None = None


@dataclass(frozen=True)
class EngineContext:
    session_phase: SessionPhase
    session_progress: float
    seasonality: Mapping[str, float]
    hidden_vol: float
    excitation: Mapping[str, float]
    burst_state: str
    shock: ShockState | None
    meta_orders: Mapping[AggressorSide, MetaOrderState | None]
    spread_excess: float
    best_depth_deficit_bid: float
    best_depth_deficit_ask: float
    imbalance_displacement: float


PARTICIPANT_TYPES: tuple[ParticipantType, ...] = (
    "passive_lp",
    "inventory_mm",
    "noise_taker",
    "informed_meta",
)

EXCITATION_KEYS: tuple[str, ...] = (
    "market_buy",
    "market_sell",
    "cancel_bid_near",
    "cancel_ask_near",
    "limit_bid_near",
    "limit_ask_near",
)

SHOCK_NAMES: tuple[ShockName, ...] = (
    "fair_jump",
    "vol_burst",
    "liquidity_drought",
    "one_sided_taker_surge",
)


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

    next_hidden_vol = hidden_vol + (params.hidden_vol_reversion * (target - hidden_vol))
    next_hidden_vol += params.hidden_vol_vol * rng.normal()
    return float(clamp(next_hidden_vol, 0.08, 3.0) * config.fair_price_vol_scale)


def advance_hidden_fair_state(
    slow_component: float,
    fast_component: float,
    jump_component: float,
    *,
    features: MarketFeatures,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    params: PresetParams,
) -> tuple[float, float, float]:
    profile = params.regimes[regime]
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.7)
    depth_signal = _bounded_signal(features.depth_imbalance, scale=0.7)
    return_signal = _bounded_signal(features.recent_return / max(features.spread_price, 0.01), scale=1.2)
    meta_signal = _meta_signal(context.meta_orders)
    shock_signal = _shock_directional_signal(context.shock)

    slow_target = (
        0.04 * flow_signal
        + 0.02 * depth_signal
        + 0.4 * meta_signal
        + 0.25 * shock_signal
    )
    slow_noise = params.slow_fair_vol * (0.35 + (0.45 * context.hidden_vol)) * rng.normal()
    next_slow = slow_component + 0.16 * (slow_target - slow_component)
    next_slow += profile.fair_drift * slow_target
    next_slow += slow_noise

    fast_target = (
        0.06 * flow_signal
        + 0.03 * depth_signal
        + 0.06 * return_signal
        + 0.12 * shock_signal
        - 0.03 * context.imbalance_displacement
    )
    fast_noise = params.fast_fair_vol * math.sqrt(context.hidden_vol) * rng.normal()
    next_fast = fast_component + params.fast_fair_reversion * (fast_target - fast_component) + fast_noise

    next_jump = jump_component * 0.45
    if context.shock is not None and context.shock.name == "fair_jump":
        jump_sign = 1.0 if context.shock.side == "buy" else -1.0
        next_jump += jump_sign * context.shock.intensity * (0.6 + 0.1 * rng.normal())
    elif rng.random() < params.fair_jump_prob:
        next_jump += rng.normal(0.0, params.fair_jump_scale * 0.55)

    return float(next_slow), float(next_fast), float(next_jump)


def decay_excitation(
    excitation: Mapping[str, float],
    *,
    config: MarketConfig,
    params: PresetParams,
) -> dict[str, float]:
    decay = clamp(params.excitation_decay, 0.35, 0.98)
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
    directional_bias = _bounded_signal(fair_gap, scale=2.7) + (0.45 * _bounded_signal(features.recent_flow_imbalance, scale=0.9))
    regime_scale = {"calm": 0.7, "directional": 1.15, "stressed": 0.95}[regime]
    spawn_scale = params.meta_spawn_prob * config.meta_order_scale * context.seasonality["meta"] * regime_scale
    if context.shock is not None and context.shock.name == "one_sided_taker_surge":
        spawn_scale *= 1.15

    for side in ("buy", "sell"):
        if updated[side] is not None:
            continue
        side_sign = 1.0 if side == "buy" else -1.0
        directional_push = max(side_sign * directional_bias, 0.0)
        spawn_prob = clamp(spawn_scale * (0.35 + (0.6 * directional_push)), 0.0, 0.12)
        if rng.random() >= spawn_prob:
            continue
        qty = max(8.0, rng.lognormal(params.meta_qty_log_mean, params.meta_qty_log_sigma))
        urgency = float(clamp(0.55 + (0.55 * directional_push) + (0.14 * rng.normal()), 0.4, 1.4))
        decay_half_life = max(6, int(rng.poisson(params.meta_duration_mean) + 4))
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
    spawn_prob = params.shock_spawn_prob * config.shock_scale * context.seasonality["shock"] * regime_scale
    spawn_prob *= 0.7 + (0.3 * clamp(context.hidden_vol, 0.0, 2.0))
    if rng.random() >= clamp(spawn_prob, 0.0, 0.18):
        return None

    shock_name = rng.choice(SHOCK_NAMES, p=np.array([0.24, 0.26, 0.26, 0.24], dtype=float))
    duration = max(3, int(rng.poisson(params.shock_duration_mean) + 2))
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


def score_limit_levels(
    side: ModelSide,
    participant_type: ParticipantType,
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext | None,
    config: MarketConfig,
    params: PresetParams,
    allow_inside: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    levels = np.arange(-1, config.book_buffer_levels + 1, dtype=int)
    scores = np.full(levels.shape, -np.inf, dtype=float)
    side_sign = 1.0 if side == "bid" else -1.0
    fair_gap_ticks = hidden_fair_tick - features.mid_tick
    fair_signal = _bounded_signal(fair_gap_ticks, scale=2.6)
    imbalance_signal = _bounded_signal(features.depth_imbalance, scale=0.75)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.75)
    profile = _participant_profile(participant_type)
    shape = _dynamic_shape_profile(
        side,
        participant_type,
        features=features,
        hidden_fair_tick=hidden_fair_tick,
        regime=regime,
        context=context,
        params=params,
    )

    for idx, level in enumerate(levels):
        if level == -1 and (not allow_inside or book.spread_ticks <= 1):
            continue

        target_tick = book.resolve_limit_tick(side, int(level))
        if target_tick is None:
            continue

        distance = 0.0 if level < 0 else float(level)
        hump = -((distance - shape["hump_center"]) ** 2) / (2.0 * max(shape["hump_sigma"], 0.5) ** 2)
        wall = -((distance - shape["wall_center"]) ** 2) / (2.0 * max(shape["wall_sigma"], 0.7) ** 2)
        score = shape["intercept"]
        score -= shape["slope"] * distance
        score -= shape["curvature"] * (distance**2)
        score += shape["hump"] * math.exp(hump)
        score += shape["wall"] * math.exp(wall)
        score += 0.65 * side_sign * params.imbalance_weight * imbalance_signal * math.exp(-distance / params.imbalance_decay)
        score += 0.65 * side_sign * params.fair_weight * fair_signal * math.exp(-distance / params.fair_decay)
        score += 0.65 * side_sign * params.flow_weight * flow_signal * math.exp(-distance / params.flow_decay)
        score += params.regimes[regime].limit_offset * profile["limit_bias"]
        score -= params.stale_penalty * min(book.level_age(side, target_tick) / 10.0, 1.5)
        score -= params.gap_penalty * max(distance - 4.0, 0.0)

        if context is not None:
            same_deficit = context.best_depth_deficit_bid if side == "bid" else context.best_depth_deficit_ask
            opposite_deficit = context.best_depth_deficit_ask if side == "bid" else context.best_depth_deficit_bid
            replenish_bonus = same_deficit * math.exp(-distance / 1.2)
            pressure_penalty = 0.1 * opposite_deficit * math.exp(-distance / 1.8)
            score += 0.45 * replenish_bonus * profile["replenish_weight"]
            score -= pressure_penalty

            if context.shock is not None:
                if context.shock.name == "liquidity_drought" and context.shock.side == side and distance <= 1.0:
                    score -= 0.9 * context.shock.intensity
                elif context.shock.name == "one_sided_taker_surge":
                    if (side == "bid" and context.shock.side == "buy") or (side == "ask" and context.shock.side == "sell"):
                        score += 0.18 * context.shock.intensity * math.exp(-distance / 2.0)
                    else:
                        score -= 0.24 * context.shock.intensity * math.exp(-distance / 1.3)

            meta_order = context.meta_orders["buy" if side == "bid" else "sell"]
            if meta_order is not None:
                progress = meta_order_progress(meta_order)
                score += 0.18 * meta_order.urgency * (1.0 - progress) * math.exp(-distance / 1.6)

            if level == -1:
                score += shape["inside_bonus"]
                score += 0.1 * profile["inside_weight"] * max(side_sign * fair_signal, 0.0)
                score += 0.12 * profile["inside_weight"] * same_deficit
                score += params.regimes[regime].inside_offset
        else:
            if level == -1:
                score += params.inside_base_bonus * profile["inside_weight"]

        scores[idx] = score

    return levels, stable_softmax(scores)


def sample_participant_events(
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    limit_budget = _sample_event_budget("limit", regime=regime, context=context, config=config, params=params, rng=rng)
    market_budget = _sample_event_budget("market", regime=regime, context=context, config=config, params=params, rng=rng)
    cancel_budget = _sample_event_budget("cancel", regime=regime, context=context, config=config, params=params, rng=rng)

    for side, side_count in _allocate_side_counts(
        {
            "bid": _aggregate_limit_side_weight(
                "bid",
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                regime=regime,
                context=context,
                params=params,
            ),
            "ask": _aggregate_limit_side_weight(
                "ask",
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                regime=regime,
                context=context,
                params=params,
            ),
        },
        limit_budget,
        rng=rng,
    ):
        events.extend(
            _sample_limit_budget_events(
                _participant_mix_specs("limit", side, context=context),
                total_events=side_count,
                book=book,
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                regime=regime,
                context=context,
                rng=rng,
                config=config,
                params=params,
            )
        )

    for side, side_count in _allocate_side_counts(
        {
            "buy": _aggregate_market_side_weight(
                "buy",
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                regime=regime,
                context=context,
                params=params,
            ),
            "sell": _aggregate_market_side_weight(
                "sell",
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                regime=regime,
                context=context,
                params=params,
            ),
        },
        market_budget,
        rng=rng,
    ):
        events.extend(
            _sample_market_budget_events(
                _participant_mix_specs("market", side, context=context),
                total_events=side_count,
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                context=context,
                rng=rng,
                params=params,
            )
        )

    cancel_side_weights = {
        side: _aggregate_cancel_side_weight(
            side,
            book=book,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            context=context,
            params=params,
        )
        for side in ("bid", "ask")
        if book.all_levels(side)
    }
    for side, side_count in _allocate_side_counts(cancel_side_weights, cancel_budget, rng=rng):
        events.extend(
            _sample_cancel_budget_events(
                _participant_mix_specs("cancel", side, context=context),
                total_events=side_count,
                book=book,
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                context=context,
                rng=rng,
                params=params,
            )
        )
    return events


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
    params: PresetParams,
) -> tuple[float, float, float, float]:
    target_best_qty = max(1.0, math.exp(params.limit_qty_log_mean) * seasonality_depth)
    best_bid_deficit = max(target_best_qty - best_bid_qty, 0.0) / target_best_qty
    best_ask_deficit = max(target_best_qty - best_ask_qty, 0.0) / target_best_qty
    spread_excess = max(spread_ticks - params.initial_spread_ticks, 0.0)
    imbalance_target = _meta_signal(meta_orders)
    imbalance_displacement = abs(depth_imbalance - imbalance_target)

    decay = math.exp(-1.0 / max(params.resiliency_half_life, 1.0))
    if shock is not None and shock.name == "liquidity_drought":
        decay = min(0.985, decay + 0.08)
    if any(meta_orders.values()):
        decay = min(0.985, decay + 0.05)

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


def _sample_event_budget(
    event_type: Literal["limit", "market", "cancel"],
    *,
    regime: RegimeName,
    context: EngineContext,
    config: MarketConfig,
    params: PresetParams,
    rng: np.random.Generator,
) -> int:
    if event_type == "limit":
        base = params.target_limit_events
        multiplier = context.seasonality["limit"] * {"calm": 0.95, "directional": 1.05, "stressed": 0.9}[regime]
        multiplier *= 1.0 + (0.08 * (context.excitation["limit_bid_near"] + context.excitation["limit_ask_near"]))
        multiplier *= 1.0 + (0.12 * (context.best_depth_deficit_bid + context.best_depth_deficit_ask))
        if context.shock is not None and context.shock.name == "liquidity_drought":
            multiplier *= max(0.65, 1.0 - (0.18 * context.shock.intensity))
        scale = config.limit_rate_scale
        hard_cap = int((params.target_limit_events * 3.0) + 8.0)
    elif event_type == "market":
        base = params.target_market_events
        multiplier = context.seasonality["market"] * {"calm": 0.9, "directional": 1.15, "stressed": 1.25}[regime]
        multiplier *= 1.0 + (0.12 * (context.excitation["market_buy"] + context.excitation["market_sell"]))
        multiplier *= 1.0 + (0.08 * abs(_meta_signal(context.meta_orders)))
        if context.shock is not None:
            if context.shock.name == "one_sided_taker_surge":
                multiplier *= 1.0 + (0.35 * context.shock.intensity)
            elif context.shock.name == "vol_burst":
                multiplier *= 1.0 + (0.1 * context.shock.intensity)
        scale = config.market_rate_scale
        hard_cap = int((params.target_market_events * 4.0) + 6.0)
    else:
        base = params.target_cancel_events
        multiplier = context.seasonality["cancel"] * {"calm": 0.95, "directional": 1.05, "stressed": 1.18}[regime]
        multiplier *= 1.0 + (0.1 * (context.excitation["cancel_bid_near"] + context.excitation["cancel_ask_near"]))
        multiplier *= 1.0 + (0.14 * context.hidden_vol)
        if context.shock is not None:
            if context.shock.name == "liquidity_drought":
                multiplier *= 1.0 + (0.28 * context.shock.intensity)
            elif context.shock.name == "vol_burst":
                multiplier *= 1.0 + (0.12 * context.shock.intensity)
        scale = config.cancel_rate_scale
        hard_cap = int((params.target_cancel_events * 2.5) + 10.0)

    lam = max(0.0, base * multiplier * scale)
    return int(min(hard_cap, rng.poisson(lam)))


def _allocate_side_counts(
    side_weights: Mapping[str, float],
    total_events: int,
    *,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    if total_events <= 0 or not side_weights:
        return []

    sides = [side for side, weight in side_weights.items() if weight > 0.0]
    if not sides:
        return []

    weights = np.array([side_weights[side] for side in sides], dtype=float)
    allocations = rng.multinomial(total_events, weights / weights.sum())
    return [(side, int(count)) for side, count in zip(sides, allocations) if count > 0]


def _participant_mix_specs(
    event_type: Literal["limit", "market", "cancel"],
    side: str,
    *,
    context: EngineContext,
 ) -> list[tuple[ParticipantType, str, float]]:
    if event_type == "limit":
        weights: dict[ParticipantType, float] = {
            "passive_lp": 0.48,
            "inventory_mm": 0.35,
            "noise_taker": 0.1,
            "informed_meta": 0.07,
        }
        same_meta = context.meta_orders["buy" if side == "bid" else "sell"]
        if same_meta is not None:
            weights["informed_meta"] += 0.08 * same_meta.urgency
            weights["passive_lp"] += 0.03
            weights["noise_taker"] = max(0.05, weights["noise_taker"] - 0.03)
        if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
            weights["inventory_mm"] += 0.1
            weights["passive_lp"] += 0.04
            weights["noise_taker"] = max(0.03, weights["noise_taker"] - 0.05)
    elif event_type == "market":
        weights = {
            "passive_lp": 0.03,
            "inventory_mm": 0.12,
            "noise_taker": 0.65,
            "informed_meta": 0.2,
        }
        same_meta = context.meta_orders[side]
        if same_meta is not None:
            weights["informed_meta"] += 0.1 * same_meta.urgency
            weights["noise_taker"] = max(0.28, weights["noise_taker"] - 0.06)
        if context.shock is not None and context.shock.name == "one_sided_taker_surge" and context.shock.side == side:
            weights["noise_taker"] += 0.12
        if context.shock is not None and context.shock.name == "vol_burst":
            weights["noise_taker"] += 0.04
            weights["inventory_mm"] += 0.03
    else:
        weights = {
            "passive_lp": 0.32,
            "inventory_mm": 0.4,
            "noise_taker": 0.15,
            "informed_meta": 0.13,
        }
        opposite_meta = context.meta_orders["buy" if side == "ask" else "sell"]
        if opposite_meta is not None:
            weights["informed_meta"] += 0.08 * opposite_meta.urgency
        if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
            weights["inventory_mm"] += 0.1
            weights["passive_lp"] += 0.04
            weights["noise_taker"] = max(0.05, weights["noise_taker"] - 0.04)

    return [(participant_type, side, max(weight, 1e-6)) for participant_type, weight in weights.items()]


def _aggregate_limit_side_weight(
    side: ModelSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    params: PresetParams,
) -> float:
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.8)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.9)
    imbalance_signal = _bounded_signal(features.depth_imbalance, scale=0.8)
    side_sign = 1.0 if side == "bid" else -1.0
    same_deficit = context.best_depth_deficit_bid if side == "bid" else context.best_depth_deficit_ask
    opposite_deficit = context.best_depth_deficit_ask if side == "bid" else context.best_depth_deficit_bid
    thinness = features.thin_bid_best if side == "bid" else features.thin_ask_best
    opposite_thinness = features.thin_ask_best if side == "bid" else features.thin_bid_best
    same_trace = context.excitation["limit_bid_near"] if side == "bid" else context.excitation["limit_ask_near"]
    opposite_trace = context.excitation["limit_ask_near"] if side == "bid" else context.excitation["limit_bid_near"]
    signed_pressure = (
        (0.08 * fair_signal)
        + (0.05 * flow_signal)
        + (0.03 * imbalance_signal)
        + (0.06 * (same_deficit - opposite_deficit))
        + (0.03 * ((opposite_thinness - thinness)))
        + (0.025 * (same_trace - opposite_trace))
    )

    log_weight = params.limit_base_log_intensity + params.regimes[regime].limit_offset
    log_weight += math.log(max(context.seasonality["limit"], 1e-6))
    log_weight += 0.12 * same_deficit
    log_weight += 0.05 * thinness
    log_weight += side_sign * signed_pressure
    log_weight -= 0.04 * context.spread_excess
    if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
        log_weight -= 0.22 * context.shock.intensity
    meta_order = context.meta_orders["buy" if side == "bid" else "sell"]
    if meta_order is not None:
        log_weight += 0.08 * meta_order.urgency * (1.0 - meta_order_progress(meta_order))
    return max(0.0, clipped_exp(log_weight, low=-3.2, high=2.4))


def _aggregate_market_side_weight(
    side: AggressorSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    params: PresetParams,
) -> float:
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.8)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.9)
    imbalance_signal = _bounded_signal(features.depth_imbalance, scale=0.85)
    sign = 1.0 if side == "buy" else -1.0
    thin_signal = _bounded_signal(features.thin_ask_best - features.thin_bid_best, scale=0.65)
    same_trace = context.excitation["market_buy"] if side == "buy" else context.excitation["market_sell"]
    opposite_trace = context.excitation["market_sell"] if side == "buy" else context.excitation["market_buy"]
    trace_signal = _bounded_signal(same_trace - opposite_trace, scale=1.6)
    signed_pressure = (
        (0.12 * fair_signal)
        + (0.09 * flow_signal)
        + (0.05 * imbalance_signal)
        + (0.06 * thin_signal)
        + (0.045 * trace_signal)
    )

    log_weight = params.market_base_log_intensity + params.regimes[regime].market_offset
    log_weight += math.log(max(context.seasonality["market"], 1e-6))
    log_weight += sign * signed_pressure
    log_weight += 0.08 * context.hidden_vol
    log_weight -= 0.14 * max(features.spread_ticks - 1, 0)

    meta_order = context.meta_orders[side]
    if meta_order is not None:
        log_weight += 0.12 * meta_order.urgency * (1.0 - meta_order_progress(meta_order))
    if context.shock is not None:
        if context.shock.name == "one_sided_taker_surge" and context.shock.side == side:
            log_weight += 0.22 * context.shock.intensity
        elif context.shock.name == "vol_burst":
            log_weight += 0.05 * context.shock.intensity
    return max(0.0, clipped_exp(log_weight, low=-3.6, high=2.0))


def _aggregate_cancel_side_weight(
    side: ModelSide,
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    params: PresetParams,
) -> float:
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.8)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.9)
    imbalance_signal = _bounded_signal(features.depth_imbalance, scale=0.85)
    sign = 1.0 if side == "ask" else -1.0
    signed_pressure = (
        (0.08 * fair_signal)
        + (0.06 * flow_signal)
        + (0.03 * imbalance_signal)
    )

    log_weight = math.log(max(context.seasonality["cancel"], 1e-6))
    log_weight += params.regimes[regime].cancel_offset
    log_weight += 0.12 * math.log1p(len(book.all_levels(side)))
    log_weight += 0.18 * context.hidden_vol
    log_weight += sign * signed_pressure
    log_weight += 0.05 * (context.excitation["cancel_ask_near"] if side == "ask" else context.excitation["cancel_bid_near"])

    if context.shock is not None:
        if context.shock.name == "liquidity_drought" and context.shock.side == side:
            log_weight += 0.24 * context.shock.intensity
        elif context.shock.name == "vol_burst":
            log_weight += 0.05 * context.shock.intensity
    return max(0.0, clipped_exp(log_weight, low=-3.2, high=2.2))


def _allocate_budget_counts(
    specs: Sequence[tuple[Any, Any, float]],
    total_events: int,
    *,
    rng: np.random.Generator,
) -> list[tuple[Any, Any, int]]:
    if total_events <= 0:
        return []

    positive_specs = [(left, right, weight) for left, right, weight in specs if weight > 0.0]
    if not positive_specs:
        return []

    weights = np.array([weight for _, _, weight in positive_specs], dtype=float)
    allocations = rng.multinomial(total_events, weights / weights.sum())
    return [
        (left, right, int(count))
        for (left, right, _), count in zip(positive_specs, allocations)
        if count > 0
    ]


def _sample_limit_budget_events(
    specs: Sequence[tuple[ParticipantType, ModelSide, float]],
    *,
    total_events: int,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for participant_type, side, count in _allocate_budget_counts(specs, total_events, rng=rng):
        levels, probabilities = score_limit_levels(
            side,
            participant_type,
            book=book,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            context=context,
            config=config,
            params=params,
        )
        chosen_levels = rng.choice(levels, size=count, replace=True, p=probabilities)
        meta_order = context.meta_orders["buy" if side == "bid" else "sell"]
        source = "meta_order" if participant_type == "informed_meta" and meta_order is not None else "organic"
        if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
            source = "shock"

        for level in chosen_levels:
            events.append(
                {
                    "type": "limit",
                    "side": side,
                    "level": int(level),
                    "qty": _sample_limit_qty(
                        participant_type,
                        side,
                        context=context,
                        rng=rng,
                        params=params,
                    ),
                    "source": source,
                    "participant_type": participant_type,
                    "meta_order_id": meta_order.id if meta_order is not None else None,
                    "meta_order_side": meta_order.side if meta_order is not None else None,
                }
            )
    return events


def _sample_market_budget_events(
    specs: Sequence[tuple[ParticipantType, AggressorSide, float]],
    *,
    total_events: int,
    features: MarketFeatures,
    hidden_fair_tick: float,
    context: EngineContext,
    rng: np.random.Generator,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for participant_type, side, count in _allocate_budget_counts(specs, total_events, rng=rng):
        meta_order = context.meta_orders.get(side)
        source = "meta_order" if participant_type == "informed_meta" and meta_order is not None else "organic"
        if context.shock is not None and context.shock.name == "one_sided_taker_surge" and context.shock.side == side:
            source = "shock"

        for _ in range(count):
            events.append(
                {
                    "type": "market",
                    "side": side,
                    "qty": _sample_market_qty(
                        side,
                        participant_type,
                        features=features,
                        hidden_fair_tick=hidden_fair_tick,
                        context=context,
                        rng=rng,
                        params=params,
                    ),
                    "source": source,
                    "participant_type": participant_type,
                    "meta_order_id": meta_order.id if meta_order is not None else None,
                    "meta_order_side": meta_order.side if meta_order is not None else None,
                }
            )
    return events


def _sample_cancel_budget_events(
    specs: Sequence[tuple[ParticipantType, ModelSide, float]],
    *,
    total_events: int,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    context: EngineContext,
    rng: np.random.Generator,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for participant_type, side, count in _allocate_budget_counts(specs, total_events, rng=rng):
        levels = book.all_levels(side)
        if not levels:
            continue

        weights = np.array(
            [
                _cancel_level_weight(
                    side,
                    depth_index=depth_index,
                    tick=tick,
                    qty=qty,
                    features=features,
                    hidden_fair_tick=hidden_fair_tick,
                    context=context,
                    book=book,
                    participant_type=participant_type,
                    params=params,
                )
                for depth_index, (tick, qty) in enumerate(levels)
            ],
            dtype=float,
        )
        chosen_indices = rng.choice(np.arange(len(levels)), size=count, replace=True, p=weights / weights.sum())
        opposite_meta = context.meta_orders["buy" if side == "ask" else "sell"]
        source = "meta_order" if participant_type == "informed_meta" and opposite_meta is not None else "organic"
        if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
            source = "shock"

        for chosen_index in chosen_indices:
            tick, qty = levels[int(chosen_index)]
            chunk = min(qty, _sample_cancel_qty(qty, participant_type=participant_type, rng=rng, params=params))
            if chunk <= 0:
                continue
            events.append(
                {
                    "type": "cancel",
                    "side": side,
                    "level": int(chosen_index),
                    "tick": tick,
                    "qty": chunk,
                    "source": source,
                    "participant_type": participant_type,
                    "meta_order_id": opposite_meta.id if opposite_meta is not None else None,
                    "meta_order_side": opposite_meta.side if opposite_meta is not None else None,
                }
            )
    return events


def _sample_limit_events_for_participant(
    participant_type: ParticipantType,
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for side in ("bid", "ask"):
        intensity = _limit_intensity(
            participant_type,
            side,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            context=context,
            config=config,
            params=params,
        )
        n_orders = min(8, int(rng.poisson(intensity)))
        if n_orders <= 0:
            continue

        levels, probabilities = score_limit_levels(
            side,
            participant_type,
            book=book,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            context=context,
            config=config,
            params=params,
        )
        chosen_levels = rng.choice(levels, size=n_orders, replace=True, p=probabilities)
        for level in chosen_levels:
            meta_order = context.meta_orders["buy" if side == "bid" else "sell"]
            source = "meta_order" if participant_type == "informed_meta" and meta_order is not None else "organic"
            if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
                source = "shock"
            events.append(
                {
                    "type": "limit",
                    "side": side,
                    "level": int(level),
                    "qty": _sample_limit_qty(
                        participant_type,
                        side,
                        context=context,
                        rng=rng,
                        params=params,
                    ),
                    "source": source,
                    "participant_type": participant_type,
                    "meta_order_id": meta_order.id if meta_order is not None else None,
                    "meta_order_side": meta_order.side if meta_order is not None else None,
                }
            )
    return events


def _sample_market_events_for_participant(
    participant_type: ParticipantType,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    intensities = {
        side: _market_intensity(
            participant_type,
            side,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            context=context,
            config=config,
            params=params,
        )
        for side in ("buy", "sell")
    }

    total_intensity = intensities["buy"] + intensities["sell"]
    if total_intensity > 0.0 and participant_type in {"noise_taker", "informed_meta"}:
        directional_edge = abs(intensities["buy"] - intensities["sell"]) / total_intensity
        if directional_edge >= 0.12:
            dominant_side = "buy" if intensities["buy"] >= intensities["sell"] else "sell"
            suppressed_side = "sell" if dominant_side == "buy" else "buy"
            suppression = 0.45 + (0.35 * directional_edge)
            intensities[suppressed_side] *= max(0.08, 1.0 - suppression)

    for side in ("buy", "sell"):
        intensity = intensities[side]
        n_orders = min(6, int(rng.poisson(intensity)))
        for _ in range(max(0, n_orders)):
            meta_order = context.meta_orders.get(side)
            source = "meta_order" if participant_type == "informed_meta" and meta_order is not None else "organic"
            if context.shock is not None and context.shock.name == "one_sided_taker_surge" and context.shock.side == side:
                source = "shock"
            events.append(
                {
                    "type": "market",
                    "side": side,
                    "qty": _sample_market_qty(
                        side,
                        participant_type,
                        features=features,
                        hidden_fair_tick=hidden_fair_tick,
                        context=context,
                        rng=rng,
                        params=params,
                    ),
                    "source": source,
                    "participant_type": participant_type,
                    "meta_order_id": meta_order.id if meta_order is not None else None,
                    "meta_order_side": meta_order.side if meta_order is not None else None,
                }
            )
    return events


def _sample_cancel_events_for_participant(
    participant_type: ParticipantType,
    *,
    book: OrderBook,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    rng: np.random.Generator,
    config: MarketConfig,
    params: PresetParams,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for side in ("bid", "ask"):
        levels = book.all_levels(side)
        if not levels:
            continue

        intensity = _cancel_intensity(
            participant_type,
            side,
            features=features,
            hidden_fair_tick=hidden_fair_tick,
            regime=regime,
            context=context,
            book=book,
            config=config,
            params=params,
        )
        n_cancels = min(max(1, len(levels)), int(rng.poisson(intensity)))
        if n_cancels <= 0:
            continue

        weights = np.array(
            [
                _cancel_level_weight(
                    side,
                    depth_index=depth_index,
                    tick=tick,
                    qty=qty,
                    features=features,
                    hidden_fair_tick=hidden_fair_tick,
                    context=context,
                    book=book,
                    participant_type=participant_type,
                    params=params,
                )
                for depth_index, (tick, qty) in enumerate(levels)
            ],
            dtype=float,
        )
        weights = weights / weights.sum()
        chosen_indices = rng.choice(np.arange(len(levels)), size=n_cancels, replace=True, p=weights)
        for chosen_index in chosen_indices:
            tick, qty = levels[int(chosen_index)]
            chunk = min(
                qty,
                _sample_cancel_qty(qty, participant_type=participant_type, rng=rng, params=params),
            )
            if chunk <= 0:
                continue
            source = "organic"
            if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
                source = "shock"
            opposite_meta = context.meta_orders["buy" if side == "ask" else "sell"]
            if participant_type == "informed_meta" and opposite_meta is not None:
                source = "meta_order"
            events.append(
                {
                    "type": "cancel",
                    "side": side,
                    "level": int(chosen_index),
                    "tick": tick,
                    "qty": chunk,
                    "source": source,
                    "participant_type": participant_type,
                    "meta_order_id": opposite_meta.id if opposite_meta is not None else None,
                    "meta_order_side": opposite_meta.side if opposite_meta is not None else None,
                }
            )
    return events


def _limit_intensity(
    participant_type: ParticipantType,
    side: ModelSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    profile = _participant_profile(participant_type)
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.6)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.75)
    imbalance_signal = _bounded_signal(features.depth_imbalance, scale=0.6)
    same_deficit = context.best_depth_deficit_bid if side == "bid" else context.best_depth_deficit_ask
    thinness = features.thin_bid_best if side == "bid" else features.thin_ask_best
    same_limit_trace = context.excitation["limit_bid_near"] if side == "bid" else context.excitation["limit_ask_near"]
    market_trace = context.excitation["market_buy"] if side == "bid" else context.excitation["market_sell"]
    side_sign = 1.0 if side == "bid" else -1.0

    log_lambda = params.limit_base_log_intensity + params.regimes[regime].limit_offset
    log_lambda += math.log(max(profile["limit"], 1e-6))
    log_lambda += math.log(max(context.seasonality["limit"], 1e-6))
    log_lambda += 0.28 * same_deficit * profile["replenish_weight"]
    log_lambda += 0.14 * thinness
    log_lambda += 0.12 * same_limit_trace * config.excitation_scale
    log_lambda += 0.08 * market_trace * config.excitation_scale
    log_lambda += 0.06 * side_sign * fair_signal * profile["directional_weight"]
    log_lambda += 0.04 * side_sign * flow_signal * profile["directional_weight"]
    log_lambda += 0.03 * side_sign * imbalance_signal * profile["directional_weight"]
    log_lambda -= 0.06 * context.hidden_vol
    log_lambda -= 0.1 * context.spread_excess

    if context.shock is not None:
        if context.shock.name == "liquidity_drought" and context.shock.side == side:
            log_lambda -= 0.55 * context.shock.intensity
        elif context.shock.name == "vol_burst":
            log_lambda -= 0.08 * context.shock.intensity

    meta_order = context.meta_orders["buy" if side == "bid" else "sell"]
    if meta_order is not None:
        log_lambda += 0.18 * meta_order.urgency * (1.0 - meta_order_progress(meta_order))

    return max(0.02, clipped_exp(log_lambda, low=-3.6, high=2.8) * config.limit_rate_scale)


def _market_intensity(
    participant_type: ParticipantType,
    aggressor_side: AggressorSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    profile = _participant_profile(participant_type)
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.3)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.75)
    imbalance_signal = _bounded_signal(features.depth_imbalance, scale=0.55)
    sign = 1.0 if aggressor_side == "buy" else -1.0
    directional = (
        max(sign * fair_signal, 0.0)
        + 0.75 * max(sign * flow_signal, 0.0)
        + 0.55 * max(sign * imbalance_signal, 0.0)
    )
    opposite_thinness = features.thin_ask_best if aggressor_side == "buy" else features.thin_bid_best
    same_market_trace = context.excitation["market_buy"] if aggressor_side == "buy" else context.excitation["market_sell"]
    opposite_market_trace = context.excitation["market_sell"] if aggressor_side == "buy" else context.excitation["market_buy"]
    cancel_trace = context.excitation["cancel_ask_near"] if aggressor_side == "buy" else context.excitation["cancel_bid_near"]

    log_lambda = params.market_base_log_intensity + params.regimes[regime].market_offset
    log_lambda += math.log(max(profile["market"], 1e-6))
    log_lambda += math.log(max(context.seasonality["market"], 1e-6))
    log_lambda += params.market_fair_weight * directional
    log_lambda += params.market_flow_weight * max(sign * flow_signal, 0.0)
    log_lambda += 0.16 * max(sign * imbalance_signal, 0.0) * profile["directional_weight"]
    log_lambda += params.market_thin_weight * opposite_thinness
    log_lambda += 0.12 * context.hidden_vol
    log_lambda += 0.28 * same_market_trace * config.excitation_scale
    log_lambda -= 0.16 * opposite_market_trace * config.excitation_scale
    log_lambda += 0.09 * cancel_trace * config.excitation_scale
    log_lambda -= params.market_spread_weight * max(features.spread_ticks - 1, 0)

    meta_order = context.meta_orders[aggressor_side]
    if meta_order is not None:
        progress_left = 1.0 - meta_order_progress(meta_order)
        log_lambda += 0.35 * meta_order.urgency * progress_left * config.meta_order_scale
    if context.shock is not None:
        if context.shock.name == "one_sided_taker_surge" and context.shock.side == aggressor_side:
            log_lambda += 0.7 * context.shock.intensity
        elif context.shock.name == "vol_burst":
            log_lambda += 0.16 * context.shock.intensity

    if participant_type == "noise_taker":
        log_lambda += 0.05 * context.hidden_vol
    return max(0.01, clipped_exp(log_lambda, low=-4.2, high=2.8) * config.market_rate_scale)


def _cancel_intensity(
    participant_type: ParticipantType,
    side: ModelSide,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext,
    book: OrderBook,
    config: MarketConfig,
    params: PresetParams,
) -> float:
    profile = _participant_profile(participant_type)
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.5)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.8)
    imbalance_signal = _bounded_signal(features.depth_imbalance, scale=0.6)
    sign = 1.0 if side == "ask" else -1.0
    adverse = (
        max(sign * fair_signal, 0.0)
        + 0.45 * max(sign * flow_signal, 0.0)
        + 0.2 * max(sign * imbalance_signal, 0.0)
    )
    cancel_trace = context.excitation["cancel_ask_near"] if side == "ask" else context.excitation["cancel_bid_near"]
    opposite_market_trace = context.excitation["market_buy"] if side == "ask" else context.excitation["market_sell"]
    depth_factor = math.log1p(len(book.all_levels(side)))

    log_lambda = math.log(max(profile["cancel"], 1e-6))
    log_lambda += math.log(max(context.seasonality["cancel"], 1e-6))
    log_lambda += params.regimes[regime].cancel_offset
    log_lambda += 0.3 * context.hidden_vol
    log_lambda += 0.18 * adverse * params.cancel_adverse_weight
    log_lambda += 0.14 * cancel_trace * config.excitation_scale
    log_lambda += 0.08 * opposite_market_trace * config.excitation_scale
    log_lambda += 0.08 * depth_factor

    if context.shock is not None:
        if context.shock.name == "liquidity_drought" and context.shock.side == side:
            log_lambda += 0.85 * context.shock.intensity
        elif context.shock.name == "vol_burst":
            log_lambda += 0.18 * context.shock.intensity

    opposite_meta = context.meta_orders["buy" if side == "ask" else "sell"]
    if opposite_meta is not None:
        log_lambda += 0.2 * opposite_meta.urgency * (1.0 - meta_order_progress(opposite_meta))

    return max(0.0, clipped_exp(log_lambda, low=-4.2, high=2.4) * config.cancel_rate_scale)


def _dynamic_shape_profile(
    side: ModelSide,
    participant_type: ParticipantType,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    regime: RegimeName,
    context: EngineContext | None,
    params: PresetParams,
) -> dict[str, float]:
    profile = _participant_profile(participant_type)
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.5)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.75)
    side_sign = 1.0 if side == "bid" else -1.0
    slope = params.base_shape_linear + profile["depth_shift"]
    curvature = params.base_shape_quadratic + (0.02 if participant_type == "inventory_mm" else 0.0)
    hump_center = params.hump_center + profile["hump_shift"]
    hump_sigma = params.hump_sigma + (0.25 if participant_type == "passive_lp" else 0.0)
    intercept = params.base_shape_intercept + (0.12 * profile["limit_bias"])
    hump = params.hump_weight * profile["hump_weight"]
    wall_center = hump_center + 1.8 + profile["wall_shift"]
    wall_sigma = 0.95 + profile["wall_sigma"]
    wall = 0.18 + profile["wall_weight"]
    inside_bonus = params.inside_base_bonus * profile["inside_weight"]

    if context is not None:
        spread_excess = context.spread_excess
        same_deficit = context.best_depth_deficit_bid if side == "bid" else context.best_depth_deficit_ask
        intercept += 0.22 * same_deficit
        slope += 0.08 * spread_excess
        wall += 0.35 * spread_excess
        inside_bonus += 0.12 * same_deficit

        meta_order = context.meta_orders["buy" if side == "bid" else "sell"]
        if meta_order is not None:
            intercept += 0.1 * meta_order.urgency
            wall += 0.06 * meta_order.urgency
            inside_bonus += 0.05 * meta_order.urgency

        if context.shock is not None:
            if context.shock.name == "liquidity_drought" and context.shock.side == side:
                intercept -= 0.55 * context.shock.intensity
                slope += 0.28 * context.shock.intensity
                wall += 0.65 * context.shock.intensity
                inside_bonus -= 0.45 * context.shock.intensity
            elif context.shock.name == "one_sided_taker_surge":
                if (side == "bid" and context.shock.side == "buy") or (side == "ask" and context.shock.side == "sell"):
                    intercept += 0.12 * context.shock.intensity
                    wall += 0.1 * context.shock.intensity
                else:
                    intercept -= 0.18 * context.shock.intensity
                    inside_bonus -= 0.1 * context.shock.intensity

        phase_adjustment = {"open": 0.12, "mid": -0.04, "close": 0.08}[context.session_phase]
        slope += phase_adjustment
        curvature += 0.02 * context.hidden_vol
        intercept += 0.02 * side_sign * fair_signal * profile["directional_weight"]
        intercept += 0.015 * side_sign * flow_signal * profile["directional_weight"]

    intercept += params.regimes[regime].limit_offset * 0.12
    return {
        "intercept": intercept,
        "slope": max(0.06, slope),
        "curvature": max(0.01, curvature),
        "hump": max(0.05, hump),
        "hump_center": max(1.0, hump_center),
        "hump_sigma": max(0.8, hump_sigma),
        "wall": max(0.0, wall),
        "wall_center": max(2.0, wall_center),
        "wall_sigma": max(0.8, wall_sigma),
        "inside_bonus": inside_bonus,
    }


def _cancel_level_weight(
    side: ModelSide,
    *,
    depth_index: int,
    tick: int,
    qty: int,
    features: MarketFeatures,
    hidden_fair_tick: float,
    context: EngineContext,
    book: OrderBook,
    participant_type: ParticipantType,
    params: PresetParams,
) -> float:
    fair_signal = _bounded_signal(hidden_fair_tick - features.mid_tick, scale=2.5)
    flow_signal = _bounded_signal(features.recent_flow_imbalance, scale=0.75)
    sign = 1.0 if side == "ask" else -1.0
    adverse = max(sign * fair_signal, 0.0) + 0.35 * max(sign * flow_signal, 0.0)
    near_best_bonus = math.exp(-depth_index / 1.2)
    stale = min(book.level_age(side, tick) / 12.0, 2.0)
    profile = _participant_profile(participant_type)
    weight = 0.35 + (0.22 * stale) + (0.18 * profile["cancel_bias"]) + (0.18 * near_best_bonus) + (0.1 * adverse)
    weight += 0.03 * math.log1p(qty)

    if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
        weight += 0.3 * context.shock.intensity * near_best_bonus

    opposite_meta = context.meta_orders["buy" if side == "ask" else "sell"]
    if opposite_meta is not None:
        weight += 0.18 * opposite_meta.urgency * near_best_bonus

    return max(weight, 1e-6)


def _sample_limit_qty(
    participant_type: ParticipantType,
    side: ModelSide,
    *,
    context: EngineContext,
    rng: np.random.Generator,
    params: PresetParams,
) -> int:
    profile = _participant_profile(participant_type)
    qty = rng.lognormal(mean=params.limit_qty_log_mean + profile["qty_shift"], sigma=params.limit_qty_log_sigma)
    same_deficit = context.best_depth_deficit_bid if side == "bid" else context.best_depth_deficit_ask
    qty *= 1.0 + (0.45 * same_deficit * profile["replenish_weight"])
    if context.session_phase == "open":
        qty *= 0.95
    elif context.session_phase == "mid":
        qty *= 1.05
    return coerce_quantity(qty)


def _sample_market_qty(
    aggressor_side: AggressorSide,
    participant_type: ParticipantType,
    *,
    features: MarketFeatures,
    hidden_fair_tick: float,
    context: EngineContext,
    rng: np.random.Generator,
    params: PresetParams,
) -> int:
    profile = _participant_profile(participant_type)
    fair_gap = hidden_fair_tick - features.mid_tick
    sign = 1.0 if aggressor_side == "buy" else -1.0
    directional_push = max(sign * fair_gap, 0.0)
    qty = rng.lognormal(mean=params.market_qty_log_mean + profile["market_qty_shift"], sigma=params.market_qty_log_sigma)
    qty *= 1.0 + (0.12 * directional_push) + (0.26 * context.hidden_vol)
    meta_order = context.meta_orders[aggressor_side]
    if meta_order is not None:
        qty *= 1.0 + (0.35 * meta_order.urgency)
    if context.shock is not None and context.shock.name == "one_sided_taker_surge" and context.shock.side == aggressor_side:
        qty *= 1.0 + (0.45 * context.shock.intensity)
    return coerce_quantity(qty)


def _sample_cancel_qty(
    current_qty: int,
    *,
    participant_type: ParticipantType,
    rng: np.random.Generator,
    params: PresetParams,
) -> int:
    profile = _participant_profile(participant_type)
    base = rng.lognormal(
        mean=max(0.0, params.limit_qty_log_mean - 0.5 + profile["cancel_qty_shift"]),
        sigma=max(0.2, params.limit_qty_log_sigma * 0.55),
    )
    chunk = min(current_qty, coerce_quantity(base))
    return int(max(chunk, 0))


def _participant_profile(participant_type: ParticipantType) -> dict[str, float]:
    profiles: dict[ParticipantType, dict[str, float]] = {
        "passive_lp": {
            "limit": 1.5,
            "market": 0.06,
            "cancel": 0.18,
            "inside_weight": 0.22,
            "limit_bias": 0.7,
            "directional_weight": 0.45,
            "replenish_weight": 1.15,
            "depth_shift": -0.08,
            "hump_shift": 0.6,
            "hump_weight": 1.2,
            "wall_shift": 0.8,
            "wall_sigma": 0.4,
            "wall_weight": 0.18,
            "cancel_bias": 0.35,
            "qty_shift": 0.08,
            "market_qty_shift": -0.12,
            "cancel_qty_shift": -0.1,
        },
        "inventory_mm": {
            "limit": 1.1,
            "market": 0.08,
            "cancel": 0.28,
            "inside_weight": 0.55,
            "limit_bias": 0.45,
            "directional_weight": 0.58,
            "replenish_weight": 1.0,
            "depth_shift": 0.02,
            "hump_shift": -0.25,
            "hump_weight": 0.9,
            "wall_shift": 0.3,
            "wall_sigma": 0.2,
            "wall_weight": 0.08,
            "cancel_bias": 0.55,
            "qty_shift": 0.0,
            "market_qty_shift": -0.04,
            "cancel_qty_shift": 0.02,
        },
        "noise_taker": {
            "limit": 0.18,
            "market": 0.46,
            "cancel": 0.06,
            "inside_weight": 0.08,
            "limit_bias": -0.1,
            "directional_weight": 0.62,
            "replenish_weight": 0.4,
            "depth_shift": 0.12,
            "hump_shift": -0.35,
            "hump_weight": 0.55,
            "wall_shift": 0.0,
            "wall_sigma": 0.0,
            "wall_weight": 0.0,
            "cancel_bias": 0.12,
            "qty_shift": -0.18,
            "market_qty_shift": 0.05,
            "cancel_qty_shift": -0.25,
        },
        "informed_meta": {
            "limit": 0.28,
            "market": 0.62,
            "cancel": 0.18,
            "inside_weight": 0.16,
            "limit_bias": 0.12,
            "directional_weight": 1.1,
            "replenish_weight": 0.6,
            "depth_shift": 0.04,
            "hump_shift": 0.1,
            "hump_weight": 0.75,
            "wall_shift": 0.2,
            "wall_sigma": 0.15,
            "wall_weight": 0.06,
            "cancel_bias": 0.5,
            "qty_shift": -0.05,
            "market_qty_shift": 0.18,
            "cancel_qty_shift": 0.08,
        },
    }
    return profiles[participant_type]


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
