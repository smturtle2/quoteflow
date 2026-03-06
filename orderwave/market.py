from __future__ import annotations

"""Public market simulator API."""

import math
from collections import deque
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

from orderwave.book import OrderBook
from orderwave.config import MarketConfig, RegimeName, coerce_config, preset_params
from orderwave.history import HistoryBuffer
from orderwave.metrics import MarketFeatures, compute_features
from orderwave.model import (
    EXCITATION_KEYS,
    EngineContext,
    MetaOrderState,
    ShockState,
    advance_hidden_fair_state,
    advance_hidden_volatility,
    advance_meta_orders,
    advance_shock_state,
    decay_excitation,
    derive_burst_state,
    meta_order_progress,
    resolve_session_phase,
    sample_next_regime,
    sample_participant_events,
    score_limit_levels,
    seasonality_multipliers,
    update_excitation_state,
    update_resiliency_state,
)
from orderwave.utils import coerce_quantity, price_to_tick, tick_to_price
from orderwave.visualization import (
    VisualHistoryRow,
    capture_visual_history_row,
    plot_market_diagnostics,
    plot_market_overview,
    plot_order_book,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class Market:
    """Order-flow-driven synthetic market simulator.

    `Market` exposes a deliberately small public API while the internal engine
    models a sparse aggregate limit order book, conditional participant flow,
    cancellations, latent meta-orders, exogenous shocks, self-excitation, and
    multi-timescale hidden fair-value dynamics.

    Examples
    --------
    >>> from orderwave import Market
    >>> market = Market(seed=42)
    >>> _ = market.gen(steps=1_000)
    >>> snapshot = market.get()
    >>> history = market.get_history()
    >>> events = market.get_event_history()
    """

    def __init__(
        self,
        init_price: float = 100.0,
        tick_size: float = 0.01,
        levels: int = 5,
        seed: int | None = None,
        config: MarketConfig | Mapping[str, object] | None = None,
    ) -> None:
        """Create a new simulator instance."""

        if tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        if levels <= 0:
            raise ValueError("levels must be positive")

        self.tick_size = float(tick_size)
        self.levels = int(levels)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.config = coerce_config(config, self.levels)
        self._params = preset_params(self.config.preset)

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer()
        self._visual_history: list[VisualHistoryRow] = []
        self._step = 0
        self._debug_last_step_events: list[dict[str, object]] = []

        init_tick = price_to_tick(init_price, self.tick_size)
        self._init_price = tick_to_price(init_tick, self.tick_size)
        self._regime: RegimeName = "calm"

        self._day = 0
        self._session_step = 0
        self._session_phase = "open"
        self._session_progress = 0.0

        self._hidden_fair_base_tick = float(init_tick)
        self._hidden_fair_slow = 0.0
        self._hidden_fair_fast = 0.0
        self._hidden_fair_jump = self._params.initial_spread_ticks / 2.0
        self._hidden_fair_tick = self._hidden_fair_base_tick + self._hidden_fair_jump
        self._hidden_vol = 0.3

        self._excitation = {key: 0.0 for key in EXCITATION_KEYS}
        self._burst_state = "calm"
        self._shock_state: ShockState | None = None
        self._meta_orders: dict[str, MetaOrderState | None] = {"buy": None, "sell": None}
        self._next_meta_order_id = 1

        self._spread_excess = 0.0
        self._best_depth_deficit_bid = 0.0
        self._best_depth_deficit_ask = 0.0
        self._imbalance_displacement = 0.0

        self._last_trade_price = self._init_price
        self._last_trade_side: str | None = None
        self._last_trade_qty = 0.0

        self._buy_flow: deque[float] = deque(maxlen=self.config.flow_window)
        self._sell_flow: deque[float] = deque(maxlen=self.config.flow_window)
        self._mid_returns: deque[float] = deque(maxlen=self.config.vol_window)
        self._buy_exec_ema = 0.0
        self._sell_exec_ema = 0.0
        self._trade_ema_alpha = 2.0 / (self.config.flow_window + 1.0)

        self._seed_initial_book(init_tick)
        self._record_current_state()

    def step(self) -> dict[str, object]:
        """Advance the simulator by one micro-batch."""

        previous_features = self._compute_features()
        previous_mid_price = previous_features.mid_price

        self._advance_session_clock()
        self._book.increment_staleness()
        self._regime = sample_next_regime(
            self._regime,
            rng=self._rng,
            config=self.config,
            params=self._params,
        )
        self._excitation = decay_excitation(
            self._excitation,
            config=self.config,
            params=self._params,
        )
        self._burst_state = derive_burst_state(self._excitation)

        latent_context = self._current_context()
        self._hidden_vol = advance_hidden_volatility(
            self._hidden_vol,
            features=previous_features,
            regime=self._regime,
            context=latent_context,
            rng=self._rng,
            config=self.config,
            params=self._params,
        )
        self._hidden_fair_slow, self._hidden_fair_fast, self._hidden_fair_jump = advance_hidden_fair_state(
            self._hidden_fair_slow,
            self._hidden_fair_fast,
            self._hidden_fair_jump,
            features=previous_features,
            regime=self._regime,
            context=self._current_context(),
            rng=self._rng,
            params=self._params,
        )
        self._hidden_fair_tick = self._hidden_fair_base_tick + self._hidden_fair_slow + self._hidden_fair_fast + self._hidden_fair_jump
        self._shock_state = advance_shock_state(
            self._shock_state,
            features=previous_features,
            regime=self._regime,
            context=self._current_context(),
            rng=self._rng,
            config=self.config,
            params=self._params,
        )
        self._meta_orders, self._next_meta_order_id = advance_meta_orders(
            self._meta_orders,
            hidden_fair_tick=self._hidden_fair_tick,
            features=previous_features,
            regime=self._regime,
            context=self._current_context(),
            rng=self._rng,
            config=self.config,
            params=self._params,
            next_meta_order_id=self._next_meta_order_id,
        )

        sampled_events = sample_participant_events(
            book=self._book,
            features=previous_features,
            hidden_fair_tick=self._hidden_fair_tick,
            regime=self._regime,
            context=self._current_context(),
            rng=self._rng,
            config=self.config,
            params=self._params,
        )
        if sampled_events:
            order = self._rng.permutation(len(sampled_events))
            sampled_events = [sampled_events[index] for index in order]

        step_buy_volume = 0.0
        step_sell_volume = 0.0
        applied_events: list[dict[str, object]] = []
        event_idx = 0

        for event in sampled_events:
            applied = self._apply_event(event, event_idx=event_idx)
            if applied is None:
                continue
            applied_events.append(applied["event_row"])
            if applied["side"] == "buy":
                step_buy_volume += applied["fill_qty"]
            elif applied["side"] == "sell":
                step_sell_volume += applied["fill_qty"]
            event_idx += 1

        self._ensure_liquidity()
        self._buy_flow.append(step_buy_volume)
        self._sell_flow.append(step_sell_volume)
        self._update_trade_strength_ema(step_buy_volume, step_sell_volume)

        current_features = self._compute_features()
        self._mid_returns.append(current_features.mid_price - previous_mid_price)

        self._step += 1
        self._debug_last_step_events = applied_events
        self._record_current_state()
        return self.get()

    def gen(self, steps: int) -> dict[str, object]:
        """Advance the simulator by ``steps`` micro-batches."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(int(steps)):
            self.step()
        return self.get()

    def get(self) -> dict[str, object]:
        """Return the current market snapshot."""

        return self._history.current()

    def get_history(self) -> pd.DataFrame:
        """Return the compact history recorded so far."""

        return self._history.dataframe()

    def get_event_history(self) -> pd.DataFrame:
        """Return the applied event log recorded so far."""

        return self._history.event_dataframe()

    def get_debug_history(self) -> pd.DataFrame:
        """Return the event-aligned latent debug history."""

        return self._history.debug_dataframe()

    def plot(
        self,
        *,
        levels: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Render the built-in market overview figure."""

        return plot_market_overview(
            self.get_history(),
            self._visual_history,
            levels=self._resolve_plot_levels(levels),
            title=title,
            figsize=figsize,
        )

    def plot_book(
        self,
        *,
        levels: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Render the current order-book snapshot on a real price axis."""

        features = self._compute_features()
        return plot_order_book(
            self._book,
            tick_size=self.tick_size,
            levels=self._resolve_plot_levels(levels),
            microprice=features.microprice,
            title=title,
            figsize=figsize,
        )

    def plot_diagnostics(
        self,
        *,
        imbalance_bins: int = 8,
        max_lag: int = 12,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Render realism-oriented diagnostics for the simulated path."""

        return plot_market_diagnostics(
            self.get_history(),
            self.get_event_history(),
            self.get_debug_history(),
            imbalance_bins=imbalance_bins,
            max_lag=max_lag,
            title=title,
            figsize=figsize,
        )

    def _seed_initial_book(self, init_tick: int) -> None:
        best_bid_tick = max(0, init_tick)
        best_ask_tick = best_bid_tick + self._params.initial_spread_ticks

        best_qty_bid = coerce_quantity(
            self._rng.lognormal(self._params.limit_qty_log_mean + 0.35, self._params.limit_qty_log_sigma)
        )
        best_qty_ask = coerce_quantity(
            self._rng.lognormal(self._params.limit_qty_log_mean + 0.35, self._params.limit_qty_log_sigma)
        )
        self._book.add_limit("bid", best_bid_tick, best_qty_bid)
        self._book.add_limit("ask", best_ask_tick, best_qty_ask)

        bootstrap_features = self._compute_features()
        seed_order_count = max(self.config.book_buffer_levels, self.levels + 4)
        for side in ("bid", "ask"):
            levels, probabilities = score_limit_levels(
                side,
                "passive_lp",
                book=self._book,
                features=bootstrap_features,
                hidden_fair_tick=self._hidden_fair_tick,
                regime=self._regime,
                context=None,
                config=self.config,
                params=self._params,
                allow_inside=False,
            )
            valid_levels = levels[levels >= 0]
            valid_probabilities = probabilities[levels >= 0]
            valid_probabilities = valid_probabilities / valid_probabilities.sum()
            allocation = self._rng.multinomial(seed_order_count, valid_probabilities)
            for level, order_count in zip(valid_levels, allocation):
                if order_count <= 0:
                    continue
                qty = coerce_quantity(
                    self._rng.lognormal(self._params.limit_qty_log_mean, self._params.limit_qty_log_sigma)
                    * order_count
                )
                self._book.apply_limit_relative(side, int(level), qty)

        self._ensure_visible_depth()

    def _apply_event(self, event: dict[str, object], *, event_idx: int) -> dict[str, object] | None:
        event_type = event["type"]
        if event_type == "limit":
            applied_tick = self._book.apply_limit_relative(
                event["side"],
                event["level"],
                event["qty"],
            )
            if applied_tick is None:
                return None
            self._ensure_two_sided_book()
            event_features = self._compute_features()
            self._update_after_event_state(
                event_type="limit",
                side=event["side"],
                level=event["level"],
                applied_qty=event["qty"],
                fill_qty=0.0,
                features=event_features,
            )
            event_row = self._build_event_row(
                event_idx=event_idx,
                event_type="limit",
                side=event["side"],
                level=event["level"],
                price=tick_to_price(applied_tick, self.tick_size),
                requested_qty=event["qty"],
                applied_qty=event["qty"],
                fill_qty=0.0,
                fills=[],
                features=event_features,
            )
            debug_row = self._build_debug_row(
                event_idx=event_idx,
                event=event,
            )
            self._history.record_event(event_row)
            self._history.record_debug(debug_row)
            return {"event_row": event_row, "side": event["side"], "fill_qty": 0.0}

        if event_type == "market":
            result = self._book.execute_market(event["side"], event["qty"])
            if result.filled_qty <= 0:
                return None
            self._last_trade_price = tick_to_price(result.last_fill_tick, self.tick_size)
            self._last_trade_side = event["side"]
            self._last_trade_qty = float(result.filled_qty)
            self._consume_meta_order(event.get("meta_order_side"), result.filled_qty)
            self._ensure_two_sided_book()
            event_features = self._compute_features()
            self._update_after_event_state(
                event_type="market",
                side=event["side"],
                level=None,
                applied_qty=result.filled_qty,
                fill_qty=result.filled_qty,
                features=event_features,
            )
            event_row = self._build_event_row(
                event_idx=event_idx,
                event_type="market",
                side=event["side"],
                level=None,
                price=tick_to_price(result.last_fill_tick, self.tick_size),
                requested_qty=event["qty"],
                applied_qty=result.filled_qty,
                fill_qty=result.filled_qty,
                fills=[
                    (tick_to_price(fill_tick, self.tick_size), float(fill_qty))
                    for fill_tick, fill_qty in result.fills
                ],
                features=event_features,
            )
            debug_row = self._build_debug_row(
                event_idx=event_idx,
                event=event,
            )
            self._history.record_event(event_row)
            self._history.record_debug(debug_row)
            return {"event_row": event_row, "side": event["side"], "fill_qty": float(result.filled_qty)}

        canceled_qty = self._book.cancel_level(event["side"], event["tick"], event["qty"])
        if canceled_qty <= 0:
            return None
        self._ensure_two_sided_book()
        event_features = self._compute_features()
        self._update_after_event_state(
            event_type="cancel",
            side=event["side"],
            level=event.get("level"),
            applied_qty=canceled_qty,
            fill_qty=0.0,
            features=event_features,
        )
        event_row = self._build_event_row(
            event_idx=event_idx,
            event_type="cancel",
            side=event["side"],
            level=event.get("level"),
            price=tick_to_price(event["tick"], self.tick_size),
            requested_qty=event["qty"],
            applied_qty=canceled_qty,
            fill_qty=0.0,
            fills=[],
            features=event_features,
        )
        debug_row = self._build_debug_row(
            event_idx=event_idx,
            event=event,
        )
        self._history.record_event(event_row)
        self._history.record_debug(debug_row)
        return {"event_row": event_row, "side": event["side"], "fill_qty": 0.0}

    def _advance_session_clock(self) -> None:
        self._session_step += 1
        if self._session_step > self.config.steps_per_day:
            self._session_step = 1
            self._day += 1
        self._session_phase, self._session_progress = resolve_session_phase(
            self._session_step,
            self.config.steps_per_day,
        )

    def _ensure_visible_depth(self) -> None:
        minimum_levels = min(self.levels, 3)
        target_qty = coerce_quantity(math.exp(self._params.limit_qty_log_mean))
        for side in ("bid", "ask"):
            current_levels = len(self._book.top_levels(side, minimum_levels))
            for level in range(current_levels, minimum_levels):
                self._book.apply_limit_relative(side, level, target_qty)

    def _ensure_liquidity(self) -> None:
        fallback_qty = coerce_quantity(math.exp(self._params.limit_qty_log_mean))
        self._ensure_two_sided_book(fallback_qty=fallback_qty)
        self._ensure_visible_depth()

    def _ensure_two_sided_book(self, *, fallback_qty: int | None = None) -> None:
        fallback_qty = coerce_quantity(math.exp(self._params.limit_qty_log_mean)) if fallback_qty is None else fallback_qty
        if self._book.best_bid_tick is None:
            reference_tick = math.floor(self._hidden_fair_tick) - 1
            if self._book.best_ask_tick is not None:
                reference_tick = min(reference_tick, self._book.best_ask_tick - 1)
            anchor_tick = max(0, int(reference_tick))
            self._book.add_limit("bid", anchor_tick, fallback_qty)

        if self._book.best_ask_tick is None:
            reference_tick = math.ceil(self._hidden_fair_tick) + 1
            if self._book.best_bid_tick is not None:
                reference_tick = max(reference_tick, self._book.best_bid_tick + 1)
            anchor_tick = int(reference_tick)
            self._book.add_limit("ask", anchor_tick, fallback_qty)

    def _consume_meta_order(self, meta_order_side: str | None, filled_qty: float) -> None:
        if meta_order_side not in {"buy", "sell"}:
            return
        current = self._meta_orders.get(meta_order_side)
        if current is None:
            return
        remaining = max(0.0, current.remaining_qty - float(filled_qty))
        if remaining <= 1.0:
            self._meta_orders[meta_order_side] = None
            return
        self._meta_orders[meta_order_side] = MetaOrderState(
            id=current.id,
            side=current.side,
            initial_qty=current.initial_qty,
            remaining_qty=remaining,
            urgency=current.urgency,
            decay_half_life=current.decay_half_life,
            age=current.age,
        )

    def _update_trade_strength_ema(self, buy_volume: float, sell_volume: float) -> None:
        alpha = self._trade_ema_alpha
        self._buy_exec_ema = ((1.0 - alpha) * self._buy_exec_ema) + (alpha * float(buy_volume))
        self._sell_exec_ema = ((1.0 - alpha) * self._sell_exec_ema) + (alpha * float(sell_volume))

    def _compute_features(self) -> MarketFeatures:
        return compute_features(
            self._book,
            tick_size=self.tick_size,
            depth_levels=self.levels,
            buy_flow=list(self._buy_flow),
            sell_flow=list(self._sell_flow),
            mid_returns=list(self._mid_returns),
            buy_exec_ema=self._buy_exec_ema,
            sell_exec_ema=self._sell_exec_ema,
        )

    def _current_context(self) -> EngineContext:
        return EngineContext(
            session_phase=self._session_phase,
            session_progress=self._session_progress,
            seasonality=seasonality_multipliers(
                self._session_phase,
                session_progress=self._session_progress,
                scale=self.config.seasonality_scale,
            ),
            hidden_vol=self._hidden_vol,
            excitation=dict(self._excitation),
            burst_state=self._burst_state,
            shock=self._shock_state,
            meta_orders=dict(self._meta_orders),
            spread_excess=self._spread_excess,
            best_depth_deficit_bid=self._best_depth_deficit_bid,
            best_depth_deficit_ask=self._best_depth_deficit_ask,
            imbalance_displacement=self._imbalance_displacement,
        )

    def _update_after_event_state(
        self,
        *,
        event_type: str,
        side: str,
        level: int | None,
        applied_qty: float,
        fill_qty: float,
        features: MarketFeatures,
    ) -> None:
        self._excitation = update_excitation_state(
            self._excitation,
            event_type=event_type,
            side=side,
            level=level,
            applied_qty=applied_qty,
            fill_qty=fill_qty,
            config=self.config,
        )
        self._burst_state = derive_burst_state(self._excitation)
        seasonality_depth = self._current_context().seasonality["depth"]
        (
            self._spread_excess,
            self._best_depth_deficit_bid,
            self._best_depth_deficit_ask,
            self._imbalance_displacement,
        ) = update_resiliency_state(
            book=self._book,
            features=features,
            seasonality_depth=seasonality_depth,
            current_spread_excess=self._spread_excess,
            current_bid_deficit=self._best_depth_deficit_bid,
            current_ask_deficit=self._best_depth_deficit_ask,
            current_imbalance_displacement=self._imbalance_displacement,
            shock=self._shock_state,
            meta_orders=self._meta_orders,
            params=self._params,
        )

    def _record_current_state(self) -> None:
        features = self._compute_features()
        snapshot = self._build_snapshot(features)
        self._history.record(
            snapshot,
            top_n_bid_qty=features.top_bid_depth,
            top_n_ask_qty=features.top_ask_depth,
            realized_vol=features.realized_vol,
            signed_flow=features.signed_flow,
        )
        self._visual_history.append(
            capture_visual_history_row(
                self._book,
                step=self._step,
                depth=self.config.book_buffer_levels,
            )
        )

    def _resolve_plot_levels(self, levels: int | None) -> int:
        resolved = self.levels if levels is None else int(levels)
        if resolved <= 0:
            raise ValueError("levels must be positive")
        return min(resolved, self.config.book_buffer_levels)

    def _build_snapshot(self, features: MarketFeatures) -> dict[str, object]:
        return {
            "step": self._step,
            "day": self._day,
            "session_step": self._session_step,
            "session_phase": self._session_phase,
            "last_price": self._last_trade_price,
            "mid_price": features.mid_price,
            "microprice": features.microprice,
            "best_bid": tick_to_price(self._book.best_bid_tick, self.tick_size),
            "best_ask": tick_to_price(self._book.best_ask_tick, self.tick_size),
            "spread": features.spread_price,
            "bids": [
                {"price": tick_to_price(tick, self.tick_size), "qty": float(qty)}
                for tick, qty in self._book.top_levels("bid", self.levels)
            ],
            "asks": [
                {"price": tick_to_price(tick, self.tick_size), "qty": float(qty)}
                for tick, qty in self._book.top_levels("ask", self.levels)
            ],
            "last_trade_side": self._last_trade_side,
            "last_trade_qty": float(self._last_trade_qty),
            "buy_aggr_volume": features.buy_aggr_volume,
            "sell_aggr_volume": features.sell_aggr_volume,
            "trade_strength": features.trade_strength,
            "depth_imbalance": features.depth_imbalance,
            "regime": self._regime,
        }

    def _build_event_row(
        self,
        *,
        event_idx: int,
        event_type: str,
        side: str,
        level: int | None,
        price: float,
        requested_qty: int,
        applied_qty: int | float,
        fill_qty: int | float,
        fills: list[tuple[float, float]],
        features: MarketFeatures,
    ) -> dict[str, object]:
        return {
            "step": self._step + 1,
            "event_idx": event_idx,
            "day": self._day,
            "session_step": self._session_step,
            "session_phase": self._session_phase,
            "event_type": event_type,
            "side": side,
            "level": level,
            "price": float(price),
            "requested_qty": float(requested_qty),
            "applied_qty": float(applied_qty),
            "fill_qty": float(fill_qty),
            "fills": list(fills),
            "best_bid_after": tick_to_price(self._book.best_bid_tick, self.tick_size),
            "best_ask_after": tick_to_price(self._book.best_ask_tick, self.tick_size),
            "mid_price_after": features.mid_price,
            "last_trade_price_after": float(self._last_trade_price),
            "regime": self._regime,
        }

    def _build_debug_row(
        self,
        *,
        event_idx: int,
        event: dict[str, object],
    ) -> dict[str, object]:
        meta_side = event.get("meta_order_side")
        meta_order = self._meta_orders.get(meta_side) if meta_side in {"buy", "sell"} else None
        return {
            "step": self._step + 1,
            "event_idx": event_idx,
            "day": self._day,
            "session_step": self._session_step,
            "session_phase": self._session_phase,
            "source": event.get("source", "organic"),
            "participant_type": event.get("participant_type"),
            "meta_order_id": event.get("meta_order_id"),
            "meta_order_side": meta_side,
            "meta_order_progress": meta_order_progress(meta_order),
            "burst_state": self._burst_state,
            "shock_state": self._shock_state.name if self._shock_state is not None else "none",
        }
