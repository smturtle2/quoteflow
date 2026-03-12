from __future__ import annotations

import math

import numpy as np

from orderwave.book import OrderBook
from orderwave.config import MarketConfig, PresetParams, RegimeName
from orderwave.metrics import MarketFeatures
from orderwave.utils import clipped_exp, stable_softmax

from .latent import _bounded_signal, meta_order_progress
from .types import AggressorSide, EngineContext, ModelSide, ParticipantType


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
        level_refill = book.level_refill_propensity(side, target_tick)
        level_revision = book.level_reprice_pressure(side, target_tick)
        level_toxicity = book.level_toxicity(side, target_tick)
        hump = -((distance - shape["hump_center"]) ** 2) / (2.0 * max(shape["hump_sigma"], 0.5) ** 2)
        wall = -((distance - shape["wall_center"]) ** 2) / (2.0 * max(shape["wall_sigma"], 0.7) ** 2)
        score = shape["intercept"]
        score -= shape["slope"] * distance
        score -= shape["curvature"] * (distance**2)
        score += shape["hump"] * math.exp(hump)
        score += shape["wall"] * math.exp(wall)
        score += 0.65 * side_sign * params.shape.imbalance_weight * imbalance_signal * math.exp(-distance / params.shape.imbalance_decay)
        score += 0.65 * side_sign * params.shape.fair_weight * fair_signal * math.exp(-distance / params.shape.fair_decay)
        score += 0.65 * side_sign * params.shape.flow_weight * flow_signal * math.exp(-distance / params.shape.flow_decay)
        score += params.regimes[regime].limit_offset * profile["limit_bias"]
        score -= params.shape.stale_penalty * min(book.level_age(side, target_tick) / 10.0, 1.5)
        score -= params.shape.gap_penalty * max(distance - 4.0, 0.0)
        score += 0.22 * level_refill * math.exp(-distance / 1.4)
        score -= 0.18 * level_revision * math.exp(-distance / 1.25)
        score -= 0.12 * level_toxicity * math.exp(-distance / 1.35)

        if context is not None:
            same_deficit = context.best_depth_deficit_bid if side == "bid" else context.best_depth_deficit_ask
            opposite_deficit = context.best_depth_deficit_ask if side == "bid" else context.best_depth_deficit_bid
            replenish_bonus = same_deficit * math.exp(-distance / 1.2)
            pressure_penalty = 0.1 * opposite_deficit * math.exp(-distance / 1.8)
            score += 0.45 * replenish_bonus * profile["replenish_weight"]
            score -= pressure_penalty
            score += 0.12 * context.recovery_pressure * math.exp(-distance / 1.4)
            score += 0.08 * context.refill_pressure * math.exp(-distance / 1.8)
            score -= 0.08 * context.passive_withdrawal * math.exp(-distance / 1.2)
            score -= 0.06 * context.maker_stress * math.exp(-distance / 1.5)
            score += 0.06 * max(side_sign * context.inventory_pressure, 0.0) * math.exp(-distance / 1.8)

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
                score += params.shape.inside_base_bonus * profile["inside_weight"]

        scores[idx] = score

    return levels, stable_softmax(scores)


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
        + (0.04 * context.directional_anchor)
        + (0.06 * (same_deficit - opposite_deficit))
        + (0.03 * ((opposite_thinness - thinness)))
        + (0.025 * (same_trace - opposite_trace))
    )

    log_weight = params.flow.limit_base_log_intensity + params.regimes[regime].limit_offset
    log_weight += math.log(max(context.seasonality["limit"], 1e-6))
    log_weight += 0.12 * same_deficit
    log_weight += 0.05 * thinness
    log_weight += 0.08 * context.recovery_pressure
    log_weight += 0.06 * context.refill_pressure
    log_weight -= 0.06 * context.passive_withdrawal
    log_weight -= 0.05 * context.maker_stress
    log_weight += side_sign * signed_pressure
    log_weight -= 0.04 * context.spread_excess
    log_weight += 0.05 * max(side_sign * context.inventory_pressure, 0.0)
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
    config: MarketConfig,
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
        + (0.14 * context.directional_anchor)
        + (0.06 * thin_signal)
        + (0.045 * trace_signal)
    )

    log_weight = params.flow.market_base_log_intensity + params.regimes[regime].market_offset
    log_weight += math.log(max(context.seasonality["market"], 1e-6))
    log_weight += sign * signed_pressure
    log_weight += 0.08 * context.hidden_vol
    log_weight -= 0.14 * max(features.spread_ticks - 1, 0)
    log_weight += 0.08 * context.impact_residue
    log_weight += 0.07 * context.flow_toxicity
    log_weight += 0.05 * context.maker_stress
    log_weight -= 0.06 * context.noise_fatigue
    log_weight -= 0.04 * context.refill_pressure
    log_weight += 0.05 * max(sign * context.inventory_pressure, 0.0)

    meta_order = context.meta_orders[side]
    if meta_order is not None:
        progress_left = 1.0 - meta_order_progress(meta_order)
        log_weight += 0.28 * meta_order.urgency * progress_left * config.meta_order_scale
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
    log_weight += 0.06 * context.drought_age
    log_weight += 0.04 * context.passive_withdrawal
    log_weight += 0.08 * context.quote_revision_pressure
    log_weight += 0.05 * context.maker_stress

    if context.shock is not None:
        if context.shock.name == "liquidity_drought" and context.shock.side == side:
            log_weight += 0.24 * context.shock.intensity
        elif context.shock.name == "vol_burst":
            log_weight += 0.05 * context.shock.intensity
    return max(0.0, clipped_exp(log_weight, low=-3.2, high=2.2))


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
    slope = params.shape.base_shape_linear + profile["depth_shift"]
    curvature = params.shape.base_shape_quadratic + (0.02 if participant_type == "inventory_mm" else 0.0)
    hump_center = params.shape.hump_center + profile["hump_shift"]
    hump_sigma = params.shape.hump_sigma + (0.25 if participant_type == "passive_lp" else 0.0)
    intercept = params.shape.base_shape_intercept + (0.12 * profile["limit_bias"])
    hump = params.shape.hump_weight * profile["hump_weight"]
    wall_center = hump_center + 1.8 + profile["wall_shift"]
    wall_sigma = 0.95 + profile["wall_sigma"]
    wall = 0.18 + profile["wall_weight"]
    inside_bonus = params.shape.inside_base_bonus * profile["inside_weight"]

    if context is not None:
        spread_excess = context.spread_excess
        same_deficit = context.best_depth_deficit_bid if side == "bid" else context.best_depth_deficit_ask
        intercept += 0.22 * same_deficit
        slope += 0.08 * spread_excess
        wall += 0.35 * spread_excess
        inside_bonus += 0.12 * same_deficit
        intercept += 0.08 * context.recovery_pressure
        intercept += 0.06 * context.refill_pressure
        intercept -= 0.06 * context.passive_withdrawal
        intercept -= 0.04 * context.maker_stress
        inside_bonus += 0.05 * max(side_sign * context.inventory_pressure, 0.0)

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
        microphase_adjustment = {
            "open_release": 0.08,
            "morning_trend": 0.03,
            "midday_lull": -0.08,
            "power_hour": 0.02,
            "closing_imbalance": 0.1,
        }[context.microphase]
        slope += phase_adjustment
        slope += microphase_adjustment
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
    revision = book.level_reprice_pressure(side, tick)
    refill = book.level_refill_propensity(side, tick)
    toxicity = book.level_toxicity(side, tick)
    profile = _participant_profile(participant_type)
    weight = 0.35 + (0.22 * stale) + (0.18 * profile["cancel_bias"]) + (0.18 * near_best_bonus) + (0.1 * adverse)
    weight += 0.03 * math.log1p(qty)
    weight += 0.18 * revision
    weight += 0.1 * toxicity
    weight -= 0.08 * refill

    if context.shock is not None and context.shock.name == "liquidity_drought" and context.shock.side == side:
        weight += 0.3 * context.shock.intensity * near_best_bonus

    opposite_meta = context.meta_orders["buy" if side == "ask" else "sell"]
    if opposite_meta is not None:
        weight += 0.18 * opposite_meta.urgency * near_best_bonus

    return max(weight, 1e-6)


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

    log_lambda = params.flow.limit_base_log_intensity + params.regimes[regime].limit_offset
    log_lambda += math.log(max(profile["limit"], 1e-6))
    log_lambda += math.log(max(context.seasonality["limit"], 1e-6))
    log_lambda += 0.28 * same_deficit * profile["replenish_weight"]
    log_lambda += 0.14 * thinness
    log_lambda += 0.12 * same_limit_trace * config.excitation_scale
    log_lambda += 0.08 * market_trace * config.excitation_scale
    log_lambda += 0.08 * context.refill_pressure
    log_lambda += 0.06 * side_sign * fair_signal * profile["directional_weight"]
    log_lambda += 0.04 * side_sign * flow_signal * profile["directional_weight"]
    log_lambda += 0.03 * side_sign * imbalance_signal * profile["directional_weight"]
    log_lambda -= 0.06 * context.hidden_vol
    log_lambda -= 0.1 * context.spread_excess
    log_lambda -= 0.08 * context.maker_stress

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
    anchor_signal = _bounded_signal(context.directional_anchor, scale=1.0)
    sign = 1.0 if aggressor_side == "buy" else -1.0
    directional = (
        max(sign * fair_signal, 0.0)
        + 0.75 * max(sign * flow_signal, 0.0)
        + 0.55 * max(sign * imbalance_signal, 0.0)
        + 0.85 * max(sign * anchor_signal, 0.0)
    )
    opposite_thinness = features.thin_ask_best if aggressor_side == "buy" else features.thin_bid_best
    same_market_trace = context.excitation["market_buy"] if aggressor_side == "buy" else context.excitation["market_sell"]
    opposite_market_trace = context.excitation["market_sell"] if aggressor_side == "buy" else context.excitation["market_buy"]
    cancel_trace = context.excitation["cancel_ask_near"] if aggressor_side == "buy" else context.excitation["cancel_bid_near"]

    log_lambda = params.flow.market_base_log_intensity + params.regimes[regime].market_offset
    log_lambda += math.log(max(profile["market"], 1e-6))
    log_lambda += math.log(max(context.seasonality["market"], 1e-6))
    log_lambda += params.flow.market_fair_weight * directional
    log_lambda += params.flow.market_flow_weight * max(sign * flow_signal, 0.0)
    log_lambda += 0.16 * max(sign * imbalance_signal, 0.0) * profile["directional_weight"]
    log_lambda += params.flow.market_thin_weight * opposite_thinness
    log_lambda += 0.12 * context.hidden_vol
    log_lambda += 0.12 * context.flow_toxicity
    log_lambda += 0.05 * context.maker_stress
    log_lambda += 0.28 * same_market_trace * config.excitation_scale
    log_lambda -= 0.16 * opposite_market_trace * config.excitation_scale
    log_lambda += 0.09 * cancel_trace * config.excitation_scale
    log_lambda -= params.flow.market_spread_weight * max(features.spread_ticks - 1, 0)

    meta_order = context.meta_orders[aggressor_side]
    if meta_order is not None:
        progress_left = 1.0 - meta_order_progress(meta_order)
        log_lambda += 0.75 * meta_order.urgency * progress_left * config.meta_order_scale
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
    log_lambda += 0.18 * adverse * params.flow.cancel_adverse_weight
    log_lambda += 0.14 * cancel_trace * config.excitation_scale
    log_lambda += 0.08 * opposite_market_trace * config.excitation_scale
    log_lambda += 0.08 * depth_factor
    log_lambda += 0.12 * context.quote_revision_pressure
    log_lambda += 0.08 * context.maker_stress

    if context.shock is not None:
        if context.shock.name == "liquidity_drought" and context.shock.side == side:
            log_lambda += 0.85 * context.shock.intensity
        elif context.shock.name == "vol_burst":
            log_lambda += 0.18 * context.shock.intensity

    opposite_meta = context.meta_orders["buy" if side == "ask" else "sell"]
    if opposite_meta is not None:
        log_lambda += 0.2 * opposite_meta.urgency * (1.0 - meta_order_progress(opposite_meta))

    return max(0.0, clipped_exp(log_lambda, low=-4.2, high=2.4) * config.cancel_rate_scale)


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
