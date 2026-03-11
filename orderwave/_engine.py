from __future__ import annotations

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
    make_cancel_log_record,
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
from orderwave.utils import EPSILON, tick_to_price

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
        market._regime = sample_next_regime(
            market._regime,
            rng=market._rng,
            config=market.config,
            params=market._params,
        )
        market._excitation = decay_excitation(
            market._excitation,
            config=market.config,
            params=market._params,
        )
        market._burst_state = derive_burst_state(market._excitation)

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
        if sampled_events:
            order = market._rng.permutation(len(sampled_events))
            sampled_events = [sampled_events[index] for index in order]
        return sampled_events

    def _apply_step_events(self, sampled_events: list[StepEvent]) -> _StepOutcome:
        step_buy_volume = 0.0
        step_sell_volume = 0.0
        event_idx = 0
        market_fill_count = 0
        for event in sampled_events:
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
        return _StepOutcome(
            sampled_event_count=len(sampled_events),
            applied_event_count=event_idx,
            step_buy_volume=step_buy_volume,
            step_sell_volume=step_sell_volume,
            market_fill_count=market_fill_count,
            liquidity_backstop_applied=False,
        )

    def _finalize_step(self, step_state: _StepState, step_outcome: _StepOutcome) -> None:
        market = self._market
        self._ensure_liquidity()
        market._buy_flow.append(step_outcome.step_buy_volume)
        market._sell_flow.append(step_outcome.step_sell_volume)
        self._update_trade_strength_ema(step_outcome.step_buy_volume, step_outcome.step_sell_volume)

        current_features = market._compute_features()
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
            debug_event["source"],
            debug_event["participant_type"],
            debug_event["meta_order_id"],
            meta_side,
            meta_order_progress(meta_order),
            market._burst_state,
            market._shock_state.name if market._shock_state is not None else "none",
        )
