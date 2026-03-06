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
    advance_hidden_fair_tick,
    sample_cancel_events,
    sample_limit_events,
    sample_market_events,
    sample_next_regime,
    score_limit_levels,
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
    models a sparse aggregate limit order book, stochastic order arrivals,
    marketable flow, cancellations, inside-spread quote improvement, a hidden
    fair-price process, and regime switching.

    Examples
    --------
    >>> from orderwave import Market
    >>> market = Market(seed=42)
    >>> _ = market.gen(steps=1_000)
    >>> snapshot = market.get()
    >>> history = market.get_history()
    """

    def __init__(
        self,
        init_price: float = 100.0,
        tick_size: float = 0.01,
        levels: int = 5,
        seed: int | None = None,
        config: MarketConfig | Mapping[str, object] | None = None,
    ) -> None:
        """Create a new simulator instance.

        Parameters
        ----------
        init_price:
            Initial reference price. It is snapped to the nearest tick.
        tick_size:
            Price increment used by the internal book.
        levels:
            Number of visible bid/ask levels returned by ``get()``.
        seed:
            Optional NumPy random seed for deterministic replay.
        config:
            Either an ``orderwave.config.MarketConfig`` instance or a plain
            mapping with the same fields.
        """

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
        self._hidden_fair_tick = init_tick + (self._params.initial_spread_ticks / 2.0)

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
        """Advance the simulator by one micro-batch.

        A step samples limit orders, marketable orders, and cancellations from
        state-conditioned distributions, shuffles those events, applies them to
        the book, records the resulting snapshot, and returns that snapshot.

        Returns
        -------
        dict[str, object]
            The latest market snapshot.
        """

        previous_features = self._compute_features()
        previous_mid_price = previous_features.mid_price

        self._book.increment_staleness()
        self._regime = sample_next_regime(
            self._regime,
            rng=self._rng,
            config=self.config,
            params=self._params,
        )
        self._hidden_fair_tick = advance_hidden_fair_tick(
            self._hidden_fair_tick,
            features=previous_features,
            regime=self._regime,
            rng=self._rng,
            config=self.config,
            params=self._params,
        )

        sampled_events = []
        sampled_events.extend(
            sample_limit_events(
                book=self._book,
                features=previous_features,
                hidden_fair_tick=self._hidden_fair_tick,
                regime=self._regime,
                rng=self._rng,
                config=self.config,
                params=self._params,
            )
        )
        sampled_events.extend(
            sample_market_events(
                features=previous_features,
                hidden_fair_tick=self._hidden_fair_tick,
                regime=self._regime,
                rng=self._rng,
                config=self.config,
                params=self._params,
            )
        )
        sampled_events.extend(
            sample_cancel_events(
                book=self._book,
                features=previous_features,
                hidden_fair_tick=self._hidden_fair_tick,
                regime=self._regime,
                rng=self._rng,
                config=self.config,
                params=self._params,
            )
        )
        if sampled_events:
            order = self._rng.permutation(len(sampled_events))
            sampled_events = [sampled_events[index] for index in order]

        step_buy_volume = 0.0
        step_sell_volume = 0.0
        applied_events: list[dict[str, object]] = []
        event_idx = 0

        for event in sampled_events:
            event_type = event["type"]
            if event_type == "limit":
                applied_tick = self._book.apply_limit_relative(
                    event["side"],
                    event["level"],
                    event["qty"],
                )
                if applied_tick is None:
                    continue
                self._ensure_two_sided_book()
                event_features = self._compute_features()
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
                self._history.record_event(event_row)
                applied_events.append(event_row)
                event_idx += 1
                continue

            if event_type == "market":
                result = self._book.execute_market(event["side"], event["qty"])
                if result.filled_qty <= 0:
                    continue
                if event["side"] == "buy":
                    step_buy_volume += result.filled_qty
                else:
                    step_sell_volume += result.filled_qty
                self._last_trade_price = tick_to_price(result.last_fill_tick, self.tick_size)
                self._last_trade_side = event["side"]
                self._last_trade_qty = float(result.filled_qty)
                self._ensure_two_sided_book()
                event_features = self._compute_features()
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
                self._history.record_event(event_row)
                applied_events.append(event_row)
                event_idx += 1
                continue

            canceled_qty = self._book.cancel_level(event["side"], event["tick"], event["qty"])
            if canceled_qty <= 0:
                continue
            self._ensure_two_sided_book()
            event_features = self._compute_features()
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
            self._history.record_event(event_row)
            applied_events.append(event_row)
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
        """Advance the simulator by ``steps`` micro-batches.

        Parameters
        ----------
        steps:
            Number of steps to execute. Must be non-negative.

        Returns
        -------
        dict[str, object]
            The latest market snapshot after the final step.
        """

        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(int(steps)):
            self.step()
        return self.get()

    def get(self) -> dict[str, object]:
        """Return the current market snapshot.

        The snapshot includes prices, spread, visible depth, recent aggressive
        flow, trade strength, depth imbalance, and the active regime.
        """

        return self._history.current()

    def get_history(self) -> pd.DataFrame:
        """Return the compact history recorded so far.

        Returns
        -------
        pandas.DataFrame
            One row per simulator step, including the initial seeded state at
            ``step == 0``.
        """

        return self._history.dataframe()

    def get_event_history(self) -> pd.DataFrame:
        """Return the applied event log recorded so far.

        Returns
        -------
        pandas.DataFrame
            One row per applied event with step-local ordering, event type,
            requested/applied quantity, fill path for market events, and the
            resulting after-state quotes.
        """

        return self._history.event_dataframe()

    def plot(
        self,
        *,
        levels: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Render the built-in market overview figure.

        Parameters
        ----------
        levels:
            Visible bid/ask depth rows to include in the heatmap. Values above
            the internal book buffer are clamped automatically.
        title:
            Optional figure title.
        figsize:
            Optional ``(width, height)`` in inches.
        """

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
        """Render the current order-book snapshot on a real price axis.

        Parameters
        ----------
        levels:
            Number of visible bid/ask levels to draw. Values above the internal
            book buffer are clamped automatically.
        title:
            Optional figure title.
        figsize:
            Optional ``(width, height)`` in inches.
        """

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
        """Render spread, imbalance, volatility, and regime diagnostics.

        Parameters
        ----------
        imbalance_bins:
            Number of bins used for the imbalance-to-next-return plot.
        max_lag:
            Maximum lag used for absolute-return autocorrelation.
        title:
            Optional figure title.
        figsize:
            Optional ``(width, height)`` in inches.
        """

        return plot_market_diagnostics(
            self.get_history(),
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
                book=self._book,
                features=bootstrap_features,
                hidden_fair_tick=self._hidden_fair_tick,
                regime=self._regime,
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
