from __future__ import annotations

"""Public market simulator API."""

import math
from collections import deque
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd

from orderwave._engine import _MarketEngine
from orderwave._model.backstop import ensure_visible_depth
from orderwave._model.latent import seasonality_multipliers
from orderwave._model.scoring import score_limit_levels
from orderwave._model.types import EXCITATION_KEYS, MetaOrderState, ShockState
from orderwave.book import OrderBook
from orderwave.config import MarketConfig, RegimeName, coerce_config, preset_params
from orderwave.history import HistoryBuffer
from orderwave.metrics import MarketFeatures, compute_features
from orderwave.utils import coerce_quantity, price_to_tick, tick_to_price
from orderwave.visualization import plot_market_diagnostics, plot_market_overview, plot_order_book

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
        self._fallback_liquidity_qty = coerce_quantity(math.exp(self._params.limit_qty_log_mean))
        self._minimum_visible_levels = min(self.levels, 3)

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer(
            logging_mode=self.config.logging_mode,
            visual_depth=self.config.book_buffer_levels,
        )
        self._visual_history = self._history.visual_store
        self._step = 0

        init_tick = price_to_tick(init_price, self.tick_size)
        self._init_price = tick_to_price(init_tick, self.tick_size)
        self._regime: RegimeName = "calm"

        self._day = 0
        self._session_step = 0
        self._session_phase = "open"
        self._session_progress = 0.0
        self._seasonality = seasonality_multipliers(
            self._session_phase,
            session_progress=self._session_progress,
            scale=self.config.seasonality_scale,
        )

        self._hidden_fair_base_tick = float(init_tick) + (self._params.initial_spread_ticks / 2.0)
        self._hidden_fair_slow = 0.0
        self._hidden_fair_fast = 0.0
        self._hidden_fair_jump = 0.0
        self._hidden_fair_tick = self._hidden_fair_base_tick
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
        self._directional_anchor = 0.0

        self._last_trade_price = self._init_price
        self._last_trade_side: str | None = None
        self._last_trade_qty = 0.0

        self._buy_flow: deque[float] = deque(maxlen=self.config.flow_window)
        self._sell_flow: deque[float] = deque(maxlen=self.config.flow_window)
        self._mid_returns: deque[float] = deque(maxlen=self.config.vol_window)
        self._buy_exec_ema = 0.0
        self._sell_exec_ema = 0.0
        self._trade_ema_alpha = 2.0 / (self.config.flow_window + 1.0)
        self._engine = _MarketEngine(self)

        self._seed_initial_book(init_tick)
        self._record_current_state(features=self._compute_features())

    def _step_impl(self) -> None:
        self._engine.run_step()

    def step(self) -> dict[str, object]:
        """Advance the simulator by one micro-batch."""

        self._step_impl()
        return self.get()

    def gen(self, steps: int) -> dict[str, object]:
        """Advance the simulator by ``steps`` micro-batches."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(int(steps)):
            self._step_impl()
        return self.get()

    def get(self) -> dict[str, object]:
        """Return the current market snapshot."""

        return self._history.current()

    def get_history(self) -> pd.DataFrame:
        """Return the compact history recorded so far."""

        return self._history.dataframe()

    def get_event_history(self) -> pd.DataFrame:
        """Return the applied event log recorded so far."""

        self._require_full_logging("event/debug logging is disabled; use logging_mode='full'")
        return self._history.event_dataframe()

    def get_debug_history(self) -> pd.DataFrame:
        """Return the event-aligned latent debug history."""

        self._require_full_logging("event/debug logging is disabled; use logging_mode='full'")
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

        self._require_full_logging("event/debug logging is disabled; use logging_mode='full'")
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

        ensure_visible_depth(
            book=self._book,
            minimum_visible_levels=self._minimum_visible_levels,
            fallback_qty=self._fallback_liquidity_qty,
        )

    def _compute_features(self) -> MarketFeatures:
        return compute_features(
            self._book,
            tick_size=self.tick_size,
            depth_levels=self.levels,
            buy_flow=self._buy_flow,
            sell_flow=self._sell_flow,
            mid_returns=self._mid_returns,
            buy_exec_ema=self._buy_exec_ema,
            sell_exec_ema=self._sell_exec_ema,
        )

    def _record_current_state(self, *, features: MarketFeatures | None = None) -> None:
        features = self._compute_features() if features is None else features
        full_bid_levels = self._book.top_levels("bid", self.config.book_buffer_levels)
        full_ask_levels = self._book.top_levels("ask", self.config.book_buffer_levels)
        snapshot = self._build_snapshot(
            features,
            bid_levels=full_bid_levels[: self.levels],
            ask_levels=full_ask_levels[: self.levels],
        )
        self._history.record(
            snapshot,
            top_n_bid_qty=features.top_bid_depth,
            top_n_ask_qty=features.top_ask_depth,
            realized_vol=features.realized_vol,
            signed_flow=features.signed_flow,
        )
        self._history.record_visual(
            step=self._step,
            bid_levels=full_bid_levels,
            ask_levels=full_ask_levels,
        )

    def _resolve_plot_levels(self, levels: int | None) -> int:
        resolved = self.levels if levels is None else int(levels)
        if resolved <= 0:
            raise ValueError("levels must be positive")
        return min(resolved, self.config.book_buffer_levels)

    def _require_full_logging(self, message: str) -> None:
        if self.config.logging_mode != "full":
            raise RuntimeError(message)

    def _build_snapshot(
        self,
        features: MarketFeatures,
        *,
        bid_levels: list[tuple[int, int]],
        ask_levels: list[tuple[int, int]],
    ) -> dict[str, object]:
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
                for tick, qty in bid_levels
            ],
            "asks": [
                {"price": tick_to_price(tick, self.tick_size), "qty": float(qty)}
                for tick, qty in ask_levels
            ],
            "last_trade_side": self._last_trade_side,
            "last_trade_qty": float(self._last_trade_qty),
            "buy_aggr_volume": features.buy_aggr_volume,
            "sell_aggr_volume": features.sell_aggr_volume,
            "trade_strength": features.trade_strength,
            "depth_imbalance": features.depth_imbalance,
            "regime": self._regime,
        }
