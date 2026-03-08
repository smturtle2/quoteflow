from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

from orderwave.book import OrderBook
from orderwave.config import MarketConfig, PresetParams, RegimeName
from orderwave.metrics import MarketFeatures
from orderwave.utils import coerce_quantity

from .events import StepEvent, make_cancel_event, make_limit_event, make_market_event
from .latent import _meta_signal
from .scoring import (
    _aggregate_cancel_side_weight,
    _aggregate_limit_side_weight,
    _aggregate_market_side_weight,
    _cancel_level_weight,
    _participant_profile,
    score_limit_levels,
)
from .types import AggressorSide, EngineContext, ModelSide, ParticipantType


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
) -> list[StepEvent]:
    events: list[StepEvent] = []
    limit_budget = _sample_event_budget("limit", regime=regime, context=context, config=config, params=params, rng=rng)
    market_budget = _sample_event_budget("market", regime=regime, context=context, config=config, params=params, rng=rng)
    cancel_budget = _sample_event_budget("cancel", regime=regime, context=context, config=config, params=params, rng=rng)

    for side, side_count in _allocate_counts(
        [
            (
                "bid",
                None,
                _aggregate_limit_side_weight(
                    "bid",
                    features=features,
                    hidden_fair_tick=hidden_fair_tick,
                    regime=regime,
                    context=context,
                    params=params,
                ),
            ),
            (
                "ask",
                None,
                _aggregate_limit_side_weight(
                    "ask",
                    features=features,
                    hidden_fair_tick=hidden_fair_tick,
                    regime=regime,
                    context=context,
                    params=params,
                ),
            ),
        ],
        limit_budget,
        rng=rng,
    ):
        events.extend(
            _sample_limit_budget_events(
                _participant_mix_specs("limit", side, context=context, config=config),
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

    for side, side_count in _allocate_counts(
        [
            (
                "buy",
                None,
                _aggregate_market_side_weight(
                    "buy",
                    features=features,
                    hidden_fair_tick=hidden_fair_tick,
                    regime=regime,
                    context=context,
                    config=config,
                    params=params,
                ),
            ),
            (
                "sell",
                None,
                _aggregate_market_side_weight(
                    "sell",
                    features=features,
                    hidden_fair_tick=hidden_fair_tick,
                    regime=regime,
                    context=context,
                    config=config,
                    params=params,
                ),
            ),
        ],
        market_budget,
        rng=rng,
    ):
        events.extend(
            _sample_market_budget_events(
                _participant_mix_specs("market", side, context=context, config=config),
                total_events=side_count,
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                context=context,
                rng=rng,
                params=params,
            )
        )

    cancel_specs = [
        (
            side,
            None,
            _aggregate_cancel_side_weight(
                side,
                book=book,
                features=features,
                hidden_fair_tick=hidden_fair_tick,
                regime=regime,
                context=context,
                params=params,
            ),
        )
        for side in ("bid", "ask")
        if book.all_levels(side)
    ]
    for side, side_count in _allocate_counts(cancel_specs, cancel_budget, rng=rng):
        events.extend(
            _sample_cancel_budget_events(
                _participant_mix_specs("cancel", side, context=context, config=config),
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
        base = params.budgets.target_limit_events
        multiplier = context.seasonality["limit"] * {"calm": 0.95, "directional": 1.05, "stressed": 0.9}[regime]
        multiplier *= 1.0 + (0.08 * (context.excitation["limit_bid_near"] + context.excitation["limit_ask_near"]))
        multiplier *= 1.0 + (0.12 * (context.best_depth_deficit_bid + context.best_depth_deficit_ask))
        if context.shock is not None and context.shock.name == "liquidity_drought":
            multiplier *= max(0.65, 1.0 - (0.18 * context.shock.intensity))
        scale = config.limit_rate_scale
        hard_cap = int((params.budgets.target_limit_events * 3.0) + 8.0)
    elif event_type == "market":
        base = params.budgets.target_market_events
        multiplier = context.seasonality["market"] * {"calm": 0.9, "directional": 1.15, "stressed": 1.25}[regime]
        multiplier *= 1.0 + (0.12 * (context.excitation["market_buy"] + context.excitation["market_sell"]))
        multiplier *= 1.0 + (0.3 * abs(_meta_signal(context.meta_orders)) * (0.7 + (0.6 * config.meta_order_scale)))
        if context.shock is not None:
            if context.shock.name == "one_sided_taker_surge":
                multiplier *= 1.0 + (0.35 * context.shock.intensity)
            elif context.shock.name == "vol_burst":
                multiplier *= 1.0 + (0.1 * context.shock.intensity)
        scale = config.market_rate_scale
        hard_cap = int((params.budgets.target_market_events * 4.0) + 6.0)
    else:
        base = params.budgets.target_cancel_events
        multiplier = context.seasonality["cancel"] * {"calm": 0.95, "directional": 1.05, "stressed": 1.18}[regime]
        multiplier *= 1.0 + (0.1 * (context.excitation["cancel_bid_near"] + context.excitation["cancel_ask_near"]))
        multiplier *= 1.0 + (0.14 * context.hidden_vol)
        if context.shock is not None:
            if context.shock.name == "liquidity_drought":
                multiplier *= 1.0 + (0.28 * context.shock.intensity)
            elif context.shock.name == "vol_burst":
                multiplier *= 1.0 + (0.12 * context.shock.intensity)
        scale = config.cancel_rate_scale
        hard_cap = int((params.budgets.target_cancel_events * 2.5) + 10.0)

    lam = max(0.0, base * multiplier * scale)
    return int(min(hard_cap, rng.poisson(lam)))


def _allocate_counts(
    specs: Sequence[tuple[Any, Any, float]],
    total_events: int,
    *,
    rng: np.random.Generator,
) -> list[tuple[Any, int]]:
    if total_events <= 0:
        return []
    positive_specs = [(left, right, weight) for left, right, weight in specs if weight > 0.0]
    if not positive_specs:
        return []
    weights = np.array([weight for _, _, weight in positive_specs], dtype=float)
    allocations = rng.multinomial(total_events, weights / weights.sum())
    if all(right is None for _, right, _ in positive_specs):
        return [(left, int(count)) for (left, _, _), count in zip(positive_specs, allocations) if count > 0]
    return [
        ((left, right), int(count))
        for (left, right, _), count in zip(positive_specs, allocations)
        if count > 0
    ]


def _participant_mix_specs(
    event_type: Literal["limit", "market", "cancel"],
    side: str,
    *,
    context: EngineContext,
    config: MarketConfig,
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
            scale_bias = max(config.meta_order_scale - 1.0, 0.0)
            weights["informed_meta"] += (0.22 + (0.4 * scale_bias)) * same_meta.urgency
            weights["noise_taker"] = max(0.12, weights["noise_taker"] - (0.12 + (0.14 * scale_bias)))
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
) -> list[StepEvent]:
    events: list[StepEvent] = []
    for participant_key, count in _allocate_counts(specs, total_events, rng=rng):
        participant_type, side = participant_key
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
                make_limit_event(
                    side=side,
                    level=int(level),
                    qty=_sample_limit_qty(
                        participant_type,
                        side,
                        context=context,
                        rng=rng,
                        params=params,
                    ),
                    source=source,
                    participant_type=participant_type,
                    meta_order_id=meta_order.id if meta_order is not None else None,
                    meta_order_side=meta_order.side if meta_order is not None else None,
                )
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
) -> list[StepEvent]:
    events: list[StepEvent] = []
    for participant_key, count in _allocate_counts(specs, total_events, rng=rng):
        participant_type, side = participant_key
        meta_order = context.meta_orders.get(side)
        source = "meta_order" if participant_type == "informed_meta" and meta_order is not None else "organic"
        if context.shock is not None and context.shock.name == "one_sided_taker_surge" and context.shock.side == side:
            source = "shock"

        for _ in range(count):
            events.append(
                make_market_event(
                    side=side,
                    qty=_sample_market_qty(
                        side,
                        participant_type,
                        features=features,
                        hidden_fair_tick=hidden_fair_tick,
                        context=context,
                        rng=rng,
                        params=params,
                    ),
                    source=source,
                    participant_type=participant_type,
                    meta_order_id=meta_order.id if meta_order is not None else None,
                    meta_order_side=meta_order.side if meta_order is not None else None,
                )
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
) -> list[StepEvent]:
    events: list[StepEvent] = []
    for participant_key, count in _allocate_counts(specs, total_events, rng=rng):
        participant_type, side = participant_key
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
                make_cancel_event(
                    side=side,
                    level=int(chosen_index),
                    tick=tick,
                    qty=chunk,
                    source=source,
                    participant_type=participant_type,
                    meta_order_id=opposite_meta.id if opposite_meta is not None else None,
                    meta_order_side=opposite_meta.side if opposite_meta is not None else None,
                )
            )
    return events


def _sample_limit_qty(
    participant_type: ParticipantType,
    side: ModelSide,
    *,
    context: EngineContext,
    rng: np.random.Generator,
    params: PresetParams,
) -> int:
    profile = _participant_profile(participant_type)
    qty = rng.lognormal(mean=params.qty.limit_qty_log_mean + profile["qty_shift"], sigma=params.qty.limit_qty_log_sigma)
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
    qty = rng.lognormal(mean=params.qty.market_qty_log_mean + profile["market_qty_shift"], sigma=params.qty.market_qty_log_sigma)
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
        mean=max(0.0, params.qty.limit_qty_log_mean - 0.5 + profile["cancel_qty_shift"]),
        sigma=max(0.2, params.qty.limit_qty_log_sigma * 0.55),
    )
    chunk = min(current_qty, coerce_quantity(base))
    return int(max(chunk, 0))
