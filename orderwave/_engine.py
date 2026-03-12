from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orderwave._model.backstop import apply_liquidity_backstop, ensure_two_sided_book
from orderwave._model.events import (
    EventLogRecord,
    EventSide,
    EventStateSnapshot,
    StepEvent,
    build_debug_event_record,
    is_cancel_event,
    is_limit_event,
    is_market_event,
    make_cancel_event,
    make_cancel_log_record,
    make_limit_event,
    make_limit_log_record,
    make_market_log_record,
)
from orderwave._model.latent import (
    advance_directional_anchor,
    advance_hidden_fair_state,
    advance_hidden_volatility,
    advance_meta_orders,
    advance_shock_state,
    decay_excitation,
    derive_burst_state,
    meta_order_progress,
    resolve_microphase,
    resolve_session_phase,
    sample_next_regime,
    seasonality_multipliers,
    update_excitation_state,
    update_resiliency_state,
)
from orderwave._model.samplers import sample_participant_events
from orderwave._model.types import EngineContext, MetaOrderState
from orderwave.config import RegimeName
from orderwave.metrics import MarketFeatures
from orderwave.utils import EPSILON, clamp, tick_to_price

if TYPE_CHECKING:
    from orderwave.market import Market


@dataclass(frozen=True)
class _StepState:
    previous_features: MarketFeatures
    previous_mid_price: float
    regime: RegimeName
    context: EngineContext
    hidden_fair_tick: float


@dataclass(frozen=True)
class _StepOutcome:
    sampled_event_count: int
    applied_event_count: int
    step_buy_volume: float
    step_sell_volume: float
    market_fill_count: int
    liquidity_backstop_applied: bool


@dataclass(frozen=True)
class _AppliedEvent:
    side: EventSide
    fill_qty: float
    event_state: EventStateSnapshot
    record: EventLogRecord


class _MarketEngine:
    def __init__(self, market: Market) -> None:
        self._market = market

    def run_step(self) -> None:
        previous_features = self._market._compute_features()
        step_state = self._advance_latent_state(previous_features)
        sampled_events = self._sample_step_events(step_state)
        step_outcome = self._apply_step_events(sampled_events)
        self._finalize_step(step_state, step_outcome)

    def _advance_latent_state(self, previous_features: MarketFeatures) -> _StepState:
        market = self._market
        previous_mid_price = previous_features.mid_price

        self._advance_session_clock()
        market._book.increment_staleness()
        market._excitation = decay_excitation(
            market._excitation,
            config=market.config,
            params=market._params,
        )
        market._burst_state = derive_burst_state(market._excitation)

        previous_regime = market._regime
        market._regime = sample_next_regime(
            market._regime,
            regime_dwell=market._regime_dwell,
            features=previous_features,
            context=self._current_context(),
            rng=market._rng,
            config=market.config,
            params=market._params,
        )
        market._regime_dwell = (market._regime_dwell + 1) if market._regime == previous_regime else 1

        latent_context = self._current_context()
        market._directional_anchor = advance_directional_anchor(
            market._directional_anchor,
            features=previous_features,
            regime=market._regime,
            context=latent_context,
            rng=market._rng,
            params=market._params,
        )
        market._hidden_vol = advance_hidden_volatility(
            market._hidden_vol,
            features=previous_features,
            regime=market._regime,
            context=self._current_context(),
            rng=market._rng,
            config=market.config,
            params=market._params,
        )
        market._hidden_fair_base_tick = previous_features.mid_tick
        market._hidden_fair_slow, market._hidden_fair_fast, market._hidden_fair_jump = advance_hidden_fair_state(
            market._hidden_fair_slow,
            market._hidden_fair_fast,
            market._hidden_fair_jump,
            features=previous_features,
            regime=market._regime,
            context=self._current_context(),
            rng=market._rng,
            config=market.config,
            params=market._params,
        )
        market._hidden_fair_tick = (
            market._hidden_fair_base_tick
            + market._hidden_fair_slow
            + market._hidden_fair_fast
            + market._hidden_fair_jump
        )
        market._shock_state = advance_shock_state(
            market._shock_state,
            features=previous_features,
            regime=market._regime,
            context=self._current_context(),
            rng=market._rng,
            config=market.config,
            params=market._params,
        )
        market._meta_orders, market._next_meta_order_id = advance_meta_orders(
            market._meta_orders,
            hidden_fair_tick=market._hidden_fair_tick,
            features=previous_features,
            regime=market._regime,
            context=self._current_context(),
            rng=market._rng,
            config=market.config,
            params=market._params,
            next_meta_order_id=market._next_meta_order_id,
        )
        self._refresh_market_state(previous_features, previous_mid_price=previous_mid_price)
        return _StepState(
            previous_features=previous_features,
            previous_mid_price=previous_mid_price,
            regime=market._regime,
            context=self._current_context(),
            hidden_fair_tick=market._hidden_fair_tick,
        )

    def _sample_step_events(self, step_state: _StepState) -> list[StepEvent]:
        market = self._market
        sampled_events = sample_participant_events(
            book=market._book,
            features=step_state.previous_features,
            hidden_fair_tick=step_state.hidden_fair_tick,
            regime=step_state.regime,
            context=step_state.context,
            rng=market._rng,
            config=market.config,
            params=market._params,
        )
        sampled_events.sort(key=self._event_stage_sort_key)
        return sampled_events

    def _apply_step_events(self, sampled_events: list[StepEvent]) -> _StepOutcome:
        pre_events = self._build_pre_withdrawal_events()
        event_idx, pre_buy, pre_sell, pre_market_fills = self._drain_events(
            pre_events,
            start_event_idx=0,
            quote_revision_wave=True,
        )
        event_idx, sampled_buy, sampled_sell, sampled_market_fills = self._drain_events(
            sampled_events,
            start_event_idx=event_idx,
            quote_revision_wave=False,
        )
        post_events = self._build_post_refill_events()
        event_idx, post_buy, post_sell, post_market_fills = self._drain_events(
            post_events,
            start_event_idx=event_idx,
            quote_revision_wave=False,
        )
        return _StepOutcome(
            sampled_event_count=len(pre_events) + len(sampled_events) + len(post_events),
            applied_event_count=event_idx,
            step_buy_volume=pre_buy + sampled_buy + post_buy,
            step_sell_volume=pre_sell + sampled_sell + post_sell,
            market_fill_count=pre_market_fills + sampled_market_fills + post_market_fills,
            liquidity_backstop_applied=False,
        )

    def _finalize_step(self, step_state: _StepState, step_outcome: _StepOutcome) -> None:
        market = self._market
        self._ensure_liquidity()
        market._buy_flow.append(step_outcome.step_buy_volume)
        market._sell_flow.append(step_outcome.step_sell_volume)
        self._update_trade_strength_ema(step_outcome.step_buy_volume, step_outcome.step_sell_volume)

        current_features = market._compute_features()
        self._refresh_market_state(current_features, previous_mid_price=step_state.previous_mid_price)
        market._mid_returns.append(current_features.mid_price - step_state.previous_mid_price)

        market._step += 1
        market._record_current_state(features=current_features)

    def _apply_event(self, event: StepEvent) -> _AppliedEvent | None:
        market = self._market

        if is_limit_event(event):
            applied_tick = market._book.apply_limit_relative(event["side"], event["level"], event["qty"])
            if applied_tick is None:
                return None
            self._ensure_two_sided_book()
            event_state = self._compute_event_state()
            self._update_after_event_state(
                event_type="limit",
                side=event["side"],
                level=event["level"],
                applied_qty=event["qty"],
                fill_qty=0.0,
                event_state=event_state,
            )
            return _AppliedEvent(
                side=event["side"],
                fill_qty=0.0,
                event_state=event_state,
                record=make_limit_log_record(
                    side=event["side"],
                    level=event["level"],
                    price=tick_to_price(applied_tick, market.tick_size),
                    qty=event["qty"],
                ),
            )

        if is_market_event(event):
            result = market._book.execute_market(event["side"], event["qty"])
            if result.filled_qty <= 0:
                return None
            market._last_trade_price = tick_to_price(result.last_fill_tick, market.tick_size)
            market._last_trade_side = event["side"]
            market._last_trade_qty = float(result.filled_qty)
            self._consume_meta_order(event["meta_order_side"], result.filled_qty)
            self._ensure_two_sided_book()
            event_state = self._compute_event_state()
            self._update_after_event_state(
                event_type="market",
                side=event["side"],
                level=None,
                applied_qty=result.filled_qty,
                fill_qty=result.filled_qty,
                event_state=event_state,
            )
            fills = tuple(
                (tick_to_price(fill_tick, market.tick_size), float(fill_qty))
                for fill_tick, fill_qty in result.fills
            )
            return _AppliedEvent(
                side=event["side"],
                fill_qty=float(result.filled_qty),
                event_state=event_state,
                record=make_market_log_record(
                    side=event["side"],
                    price=tick_to_price(result.last_fill_tick, market.tick_size),
                    requested_qty=event["qty"],
                    applied_qty=result.filled_qty,
                    fills=fills,
                ),
            )

        if is_cancel_event(event):
            canceled_qty = market._book.cancel_level(event["side"], event["tick"], event["qty"])
            if canceled_qty <= 0:
                return None
            self._ensure_two_sided_book()
            event_state = self._compute_event_state()
            self._update_after_event_state(
                event_type="cancel",
                side=event["side"],
                level=event["level"],
                applied_qty=canceled_qty,
                fill_qty=0.0,
                event_state=event_state,
            )
            return _AppliedEvent(
                side=event["side"],
                fill_qty=0.0,
                event_state=event_state,
                record=make_cancel_log_record(
                    side=event["side"],
                    level=event["level"],
                    price=tick_to_price(event["tick"], market.tick_size),
                    requested_qty=event["qty"],
                    applied_qty=canceled_qty,
                ),
            )

        return None

    def _advance_session_clock(self) -> None:
        market = self._market
        market._session_step += 1
        if market._session_step > market.config.steps_per_day:
            market._session_step = 1
            market._day += 1
        market._session_phase, market._session_progress = resolve_session_phase(
            market._session_step,
            market.config.steps_per_day,
        )
        market._microphase = resolve_microphase(market._session_progress)
        market._seasonality = seasonality_multipliers(
            market._session_phase,
            session_progress=market._session_progress,
            scale=market.config.seasonality_scale,
        )

    def _ensure_liquidity(self) -> None:
        market = self._market
        apply_liquidity_backstop(
            book=market._book,
            mode=market.config.liquidity_backstop,
            minimum_visible_levels=market._minimum_visible_levels,
            fallback_qty=market._fallback_liquidity_qty,
            hidden_fair_tick=market._hidden_fair_tick,
        )

    def _ensure_two_sided_book(self, *, fallback_qty: int | None = None) -> None:
        market = self._market
        ensure_two_sided_book(
            book=market._book,
            hidden_fair_tick=market._hidden_fair_tick,
            fallback_qty=market._fallback_liquidity_qty if fallback_qty is None else fallback_qty,
        )

    def _consume_meta_order(self, meta_order_side: str | None, filled_qty: float) -> None:
        market = self._market
        if meta_order_side not in {"buy", "sell"}:
            return
        current = market._meta_orders.get(meta_order_side)
        if current is None:
            return
        remaining = max(0.0, current.remaining_qty - float(filled_qty))
        if remaining <= 1.0:
            market._meta_orders[meta_order_side] = None
            return
        market._meta_orders[meta_order_side] = MetaOrderState(
            id=current.id,
            side=current.side,
            initial_qty=current.initial_qty,
            remaining_qty=remaining,
            urgency=current.urgency,
            decay_half_life=current.decay_half_life,
            age=current.age,
        )

    def _update_trade_strength_ema(self, buy_volume: float, sell_volume: float) -> None:
        market = self._market
        alpha = market._trade_ema_alpha
        market._buy_exec_ema = ((1.0 - alpha) * market._buy_exec_ema) + (alpha * float(buy_volume))
        market._sell_exec_ema = ((1.0 - alpha) * market._sell_exec_ema) + (alpha * float(sell_volume))

    def _current_context(self) -> EngineContext:
        market = self._market
        return EngineContext(
            session_phase=market._session_phase,
            microphase=market._microphase,
            session_progress=market._session_progress,
            seasonality=market._seasonality,
            hidden_vol=market._hidden_vol,
            excitation=market._excitation,
            burst_state=market._burst_state,
            shock=market._shock_state,
            meta_orders=market._meta_orders,
            spread_excess=market._spread_excess,
            best_depth_deficit_bid=market._best_depth_deficit_bid,
            best_depth_deficit_ask=market._best_depth_deficit_ask,
            imbalance_displacement=market._imbalance_displacement,
            directional_anchor=market._directional_anchor,
            drought_age=market._drought_age,
            recovery_pressure=market._recovery_pressure,
            one_sided_pressure=market._one_sided_pressure,
            impact_residue=market._impact_residue,
            regime_dwell=market._regime_dwell,
            inventory_pressure=market._inventory_pressure,
            passive_withdrawal=market._passive_withdrawal_bias,
            noise_fatigue=market._noise_fatigue,
            flow_toxicity=market._flow_toxicity,
            maker_stress=market._maker_stress,
            quote_revision_pressure=market._quote_revision_pressure,
            refill_pressure=market._refill_pressure,
        )

    def _update_after_event_state(
        self,
        *,
        event_type: str,
        side: str,
        level: int | None,
        applied_qty: float,
        fill_qty: float,
        event_state: EventStateSnapshot,
    ) -> None:
        market = self._market
        market._excitation = update_excitation_state(
            market._excitation,
            event_type=event_type,
            side=side,
            level=level,
            applied_qty=applied_qty,
            fill_qty=fill_qty,
            config=market.config,
        )
        market._burst_state = derive_burst_state(market._excitation)
        seasonality_depth = market._seasonality["depth"]
        (
            market._spread_excess,
            market._best_depth_deficit_bid,
            market._best_depth_deficit_ask,
            market._imbalance_displacement,
        ) = update_resiliency_state(
            seasonality_depth=seasonality_depth,
            current_spread_excess=market._spread_excess,
            current_bid_deficit=market._best_depth_deficit_bid,
            current_ask_deficit=market._best_depth_deficit_ask,
            current_imbalance_displacement=market._imbalance_displacement,
            shock=market._shock_state,
            meta_orders=market._meta_orders,
            spread_ticks=int(event_state["spread_ticks"]),
            depth_imbalance=event_state["depth_imbalance"],
            best_bid_qty=int(event_state["best_bid_qty"]),
            best_ask_qty=int(event_state["best_ask_qty"]),
            config=market.config,
            params=market._params,
        )

    def _compute_event_state(self) -> EventStateSnapshot:
        market = self._market
        best_bid_qty, best_ask_qty, bid_depth, ask_depth = market._book.top_depth_state(market.levels)
        mid_price = tick_to_price((market._book.best_bid_tick + market._book.best_ask_tick) / 2.0, market.tick_size)
        depth_imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, EPSILON)
        return {
            "mid_price": float(mid_price),
            "spread_ticks": float(market._book.spread_ticks),
            "depth_imbalance": float(depth_imbalance),
            "best_bid_qty": float(best_bid_qty),
            "best_ask_qty": float(best_ask_qty),
        }

    def _record_applied_event(
        self,
        *,
        event_idx: int,
        record: EventLogRecord,
        event_state: EventStateSnapshot,
    ) -> None:
        market = self._market
        market._history.record_event(
            market._step + 1,
            event_idx,
            market._day,
            market._session_step,
            market._session_phase,
            record["event_type"],
            record["side"],
            record["level"],
            record["price"],
            record["requested_qty"],
            record["applied_qty"],
            record["fill_qty"],
            record["fills"],
            tick_to_price(market._book.best_bid_tick, market.tick_size),
            tick_to_price(market._book.best_ask_tick, market.tick_size),
            event_state["mid_price"],
            market._last_trade_price,
            market._regime,
        )

    def _record_debug_event(self, event_idx: int, event: StepEvent) -> None:
        market = self._market
        debug_event = build_debug_event_record(event)
        meta_side = debug_event["meta_order_side"]
        meta_order = market._meta_orders.get(meta_side) if meta_side in {"buy", "sell"} else None
        market._history.record_debug(
            market._step + 1,
            event_idx,
            market._day,
            market._session_step,
            market._session_phase,
            market._microphase,
            debug_event["source"],
            debug_event["participant_type"],
            debug_event["meta_order_id"],
            meta_side,
            meta_order_progress(meta_order),
            market._burst_state,
            market._shock_state.name if market._shock_state is not None else "none",
            market._drought_age,
            market._recovery_pressure,
            market._impact_residue,
            market._regime_dwell,
            market._inventory_pressure,
            market._flow_toxicity,
            market._maker_stress,
            market._quote_revision_wave,
            market._refill_pressure,
            len(market._book.top_levels("bid", market.levels)),
            len(market._book.top_levels("ask", market.levels)),
        )

    def _event_stage_sort_key(self, event: StepEvent) -> tuple[int, int, int, int]:
        source_rank = {"shock": 0, "meta_order": 1, "organic": 2}.get(event["source"], 3)
        participant_type = event["participant_type"]
        if is_market_event(event):
            participant_rank = {
                "informed_meta": 0,
                "noise_taker": 1,
                "inventory_mm": 2,
                "passive_lp": 3,
            }[participant_type]
            return (0, participant_rank, source_rank, 0)
        if is_cancel_event(event):
            participant_rank = {
                "inventory_mm": 0,
                "informed_meta": 1,
                "passive_lp": 2,
                "noise_taker": 3,
            }[participant_type]
            level = event["level"]
            return (1, participant_rank, source_rank, 0 if level is None else int(level))
        assert is_limit_event(event)
        participant_rank = {
            "inventory_mm": 0,
            "informed_meta": 1,
            "passive_lp": 2,
            "noise_taker": 3,
        }[participant_type]
        return (2, participant_rank, source_rank, int(event["level"]))

    def _refresh_market_state(self, features: MarketFeatures, *, previous_mid_price: float) -> None:
        market = self._market
        visible_levels_bid = len(market._book.top_levels("bid", market._minimum_visible_levels))
        visible_levels_ask = len(market._book.top_levels("ask", market._minimum_visible_levels))
        visible_gap = max(market._minimum_visible_levels - visible_levels_bid, market._minimum_visible_levels - visible_levels_ask, 0)
        thinness = max(features.thin_bid_best, features.thin_ask_best)
        meta_pressure = 0.0
        for side, meta_order in market._meta_orders.items():
            if meta_order is None:
                continue
            sign = 1.0 if side == "buy" else -1.0
            meta_pressure += sign * (1.0 - meta_order_progress(meta_order)) * meta_order.urgency

        market._one_sided_pressure = float(
            clamp(
                (0.72 * market._one_sided_pressure)
                + (market.config.depletion_scale * (visible_gap / max(market._minimum_visible_levels, 1))),
                0.0,
                4.0,
            )
        )
        drought_trigger = (
            visible_gap > 0
            or market._spread_excess > 0.35
            or max(market._best_depth_deficit_bid, market._best_depth_deficit_ask) > 0.25
            or thinness > 0.35
        )
        if drought_trigger:
            market._drought_age = float(clamp(market._drought_age + market.config.depletion_scale, 0.0, 20.0))
        else:
            market._drought_age = float(max(0.0, market._drought_age - (0.6 * market.config.resiliency_scale)))

        recovery_target = (
            market._spread_excess
            + market._best_depth_deficit_bid
            + market._best_depth_deficit_ask
            + (0.5 * market._one_sided_pressure)
        )
        market._recovery_pressure = float(clamp(recovery_target * market.config.depletion_scale, 0.0, 6.0))

        return_move = abs(features.mid_price - previous_mid_price) / max(features.spread_price, market.tick_size)
        shock_pressure = 0.0 if market._shock_state is None else float(market._shock_state.intensity)
        impact_target = (
            (0.45 * abs(features.trade_strength))
            + (0.35 * return_move)
            + (0.2 * abs(meta_pressure))
            + (0.18 * shock_pressure)
        )
        market._impact_residue = float(clamp((0.78 * market._impact_residue) + impact_target, 0.0, 6.0))

        inventory_target = (
            features.signed_flow
            + (0.45 * features.depth_imbalance)
            + (0.3 * market._directional_anchor)
            + (0.25 * meta_pressure)
        )
        market._inventory_pressure = float(
            clamp(
                (0.84 * market._inventory_pressure)
                + (market.config.participant_feedback_scale * inventory_target),
                -3.0,
                3.0,
            )
        )

        withdrawal_target = (0.45 * market._recovery_pressure) + (0.35 * thinness) + (0.2 * shock_pressure)
        market._passive_withdrawal_bias = float(
            clamp(
                (0.76 * market._passive_withdrawal_bias)
                + (market.config.participant_feedback_scale * withdrawal_target),
                0.0,
                4.0,
            )
        )

        flow_activity = (features.buy_aggr_volume + features.sell_aggr_volume) / max(market.config.flow_window, 1)
        market._noise_fatigue = float(
            clamp(
                (0.74 * market._noise_fatigue)
                + (0.35 * flow_activity)
                + (0.16 * market._impact_residue)
                - (0.08 * market.config.resiliency_scale),
                0.0,
                4.0,
            )
        )

        microphase_bias = {
            "open_release": 0.18,
            "morning_trend": 0.1,
            "midday_lull": -0.08,
            "power_hour": 0.12,
            "closing_imbalance": 0.2,
        }[market._microphase]
        market._flow_toxicity = float(
            clamp(
                (0.76 * market._flow_toxicity)
                + (0.32 * abs(features.trade_strength))
                + (0.16 * abs(features.signed_flow))
                + (0.14 * thinness)
                + (0.14 * shock_pressure)
                + (0.1 * abs(market._directional_anchor)),
                0.0,
                6.0,
            )
        )
        maker_stress_target = (
            (0.34 * market._flow_toxicity)
            + (0.24 * market._recovery_pressure)
            + (0.17 * market._one_sided_pressure)
            + (0.14 * market._impact_residue)
            + (0.1 * shock_pressure)
            + max(microphase_bias, 0.0)
        )
        market._maker_stress = float(
            clamp(
                (0.74 * market._maker_stress)
                + (market._params.resiliency.revision_sensitivity * maker_stress_target)
                - (0.08 * market.config.resiliency_scale),
                0.0,
                6.0,
            )
        )
        revision_target = market._params.resiliency.withdrawal_sensitivity * (
            (0.4 * market._maker_stress)
            + (0.22 * market._flow_toxicity)
            + (0.18 * market._passive_withdrawal_bias)
            + (0.1 * shock_pressure)
            + max(microphase_bias, 0.0)
        )
        market._quote_revision_pressure = float(
            clamp((0.72 * market._quote_revision_pressure) + revision_target, 0.0, 6.0)
        )
        refill_target = market._params.resiliency.refill_strength * (
            (0.42 * market._recovery_pressure)
            + (0.18 * (market._best_depth_deficit_bid + market._best_depth_deficit_ask))
            + (0.16 * market._one_sided_pressure)
            + (0.08 * market._quote_revision_pressure)
            + (0.06 * max(-microphase_bias, 0.0))
        )
        market._refill_pressure = float(clamp((0.7 * market._refill_pressure) + refill_target, 0.0, 6.0))

    def _drain_events(
        self,
        events: list[StepEvent],
        *,
        start_event_idx: int,
        quote_revision_wave: bool,
    ) -> tuple[int, float, float, int]:
        market = self._market
        step_buy_volume = 0.0
        step_sell_volume = 0.0
        market_fill_count = 0
        event_idx = start_event_idx
        market._quote_revision_wave = quote_revision_wave
        for event in events:
            applied = self._apply_event(event)
            if applied is None:
                continue
            self._record_applied_event(event_idx=event_idx, record=applied.record, event_state=applied.event_state)
            self._record_debug_event(event_idx, event)
            if applied.side == "buy":
                step_buy_volume += applied.fill_qty
            elif applied.side == "sell":
                step_sell_volume += applied.fill_qty
            if applied.record["event_type"] == "market" and applied.fill_qty > 0.0:
                market_fill_count += 1
            event_idx += 1
        market._quote_revision_wave = False
        return event_idx, step_buy_volume, step_sell_volume, market_fill_count

    def _build_pre_withdrawal_events(self) -> list[StepEvent]:
        market = self._market
        pressure = market._quote_revision_pressure * market._params.resiliency.withdrawal_sensitivity
        if pressure < 0.45:
            return []

        features = market._compute_features()
        directional_pressure = self._directional_pressure(features)
        sides: tuple[EventSide, ...]
        if market._shock_state is not None and market._shock_state.name == "liquidity_drought" and market._shock_state.side in {
            "bid",
            "ask",
        }:
            sides = (market._shock_state.side,)
        elif directional_pressure > 0.06:
            sides = ("ask",)
        elif directional_pressure < -0.06:
            sides = ("bid",)
        else:
            sides = ()

        events: list[StepEvent] = []
        for side in sides:
            levels = market._book.top_levels(side, 2)
            for depth_index, (tick, qty) in enumerate(levels):
                toxicity = market._book.level_toxicity(side, tick)
                revision = market._book.level_reprice_pressure(side, tick)
                share = (0.06 + (0.035 * pressure) + (0.025 * toxicity) + (0.02 * revision)) * math.exp(
                    -depth_index / 1.4
                )
                canceled = min(qty - 1, max(0, int(round(qty * min(0.55, share)))))
                if canceled <= 0:
                    continue
                events.append(
                    make_cancel_event(
                        side=side,
                        level=depth_index,
                        tick=tick,
                        qty=canceled,
                        source="shock" if market._shock_state is not None and market._shock_state.side == side else "organic",
                        participant_type="inventory_mm" if depth_index == 0 else "passive_lp",
                        meta_order_id=None,
                        meta_order_side=None,
                    )
                )
        return events

    def _build_post_refill_events(self) -> list[StepEvent]:
        market = self._market
        refill_pressure = market._refill_pressure * market._params.resiliency.refill_strength
        if refill_pressure < 0.35:
            return []

        features = market._compute_features()
        directional_pressure = self._directional_pressure(features)
        events: list[StepEvent] = []
        for side in ("bid", "ask"):
            side_deficit = market._best_depth_deficit_bid if side == "bid" else market._best_depth_deficit_ask
            thinness = features.thin_bid_best if side == "bid" else features.thin_ask_best
            visible_levels = len(market._book.top_levels(side, market._minimum_visible_levels))
            visible_gap = max(market._minimum_visible_levels - visible_levels, 0) / max(market._minimum_visible_levels, 1)
            if side_deficit < 0.08 and thinness < 0.12 and visible_gap <= 0.0:
                continue

            side_sign = 1.0 if side == "bid" else -1.0
            adverse_bias = max(side_sign * directional_pressure, 0.0)
            qty_scale = max(0.55, 1.0 - (0.08 * market._flow_toxicity) - (0.05 * adverse_bias))
            refill_events = 1 + int(side_deficit > 0.2 or visible_gap > 0.0)
            if market._microphase in {"power_hour", "closing_imbalance"} and side_deficit > 0.25:
                refill_events += 1
            for refill_index in range(refill_events):
                improve_quote = (
                    refill_index == 0
                    and market._book.spread_ticks > 1
                    and side_deficit > 0.18
                    and market._maker_stress < 3.8
                )
                qty = max(
                    1,
                    int(
                        round(
                            market._fallback_liquidity_qty
                            * (0.75 + (0.28 * refill_pressure) + (0.22 * side_deficit) + (0.18 * visible_gap))
                            * qty_scale
                        )
                    ),
                )
                events.append(
                    make_limit_event(
                        side=side,
                        level=-1 if improve_quote else min(refill_index, max(0, market.levels - 1)),
                        qty=qty,
                        source="organic",
                        participant_type="inventory_mm" if refill_index == 0 else "passive_lp",
                        meta_order_id=None,
                        meta_order_side=None,
                    )
                )
        return events

    def _directional_pressure(self, features: MarketFeatures) -> float:
        market = self._market
        pressure = (
            (0.46 * market._directional_anchor)
            + (0.24 * features.signed_flow)
            + (0.18 * features.trade_strength)
            + (0.12 * market._inventory_pressure)
        )
        if market._shock_state is not None and market._shock_state.side in {"buy", "sell"}:
            pressure += 0.18 * (1.0 if market._shock_state.side == "buy" else -1.0) * market._shock_state.intensity
        return float(clamp(pressure, -2.0, 2.0))
