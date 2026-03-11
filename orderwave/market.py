from __future__ import annotations

"""Public market simulator API."""

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from orderwave._engine import _MarketEngine
from orderwave._model.backstop import ensure_visible_depth
from orderwave._model.latent import seasonality_multipliers
from orderwave._model.scoring import score_limit_levels
from orderwave._model.types import EXCITATION_KEYS, MetaOrderState, ShockState
from orderwave.book import OrderBook
from orderwave.config import (
    LiquidityBackstopMode,
    LoggingMode,
    MarketConfig,
    PresetName,
    RegimeName,
    coerce_config,
    preset_params,
)
from orderwave.history import HistoryBuffer
from orderwave.metrics import MarketFeatures, compute_features
from orderwave.utils import coerce_quantity, price_to_tick, tick_to_price
from orderwave.visualization import plot_market_diagnostics, plot_market_overview, plot_order_book

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class BookLevel:
    """Visible aggregate depth level returned by the public snapshot view."""

    price: float
    qty: float

    def to_dict(self) -> dict[str, float]:
        return {"price": float(self.price), "qty": float(self.qty)}


@dataclass(frozen=True)
class MarketSnapshot:
    """Typed view of the current public market snapshot."""

    step: int
    day: int
    session_step: int
    session_phase: str
    last_price: float
    mid_price: float
    microprice: float
    best_bid: float
    best_ask: float
    spread: float
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    last_trade_side: str | None
    last_trade_qty: float
    buy_aggr_volume: float
    sell_aggr_volume: float
    trade_strength: float
    depth_imbalance: float
    regime: RegimeName

    def to_dict(self) -> dict[str, object]:
        return {
            "step": int(self.step),
            "day": int(self.day),
            "session_step": int(self.session_step),
            "session_phase": self.session_phase,
            "last_price": float(self.last_price),
            "mid_price": float(self.mid_price),
            "microprice": float(self.microprice),
            "best_bid": float(self.best_bid),
            "best_ask": float(self.best_ask),
            "spread": float(self.spread),
            "bids": [level.to_dict() for level in self.bids],
            "asks": [level.to_dict() for level in self.asks],
            "last_trade_side": self.last_trade_side,
            "last_trade_qty": float(self.last_trade_qty),
            "buy_aggr_volume": float(self.buy_aggr_volume),
            "sell_aggr_volume": float(self.sell_aggr_volume),
            "trade_strength": float(self.trade_strength),
            "depth_imbalance": float(self.depth_imbalance),
            "regime": self.regime,
        }


@dataclass(frozen=True)
class SimulationResult:
    """Collected outputs returned by ``Market.run``."""

    snapshot: MarketSnapshot
    history: pd.DataFrame
    event_history: pd.DataFrame | None
    debug_history: pd.DataFrame | None
    labeled_event_history: pd.DataFrame | None


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
        *,
        preset: PresetName | None = None,
        logging_mode: LoggingMode | None = None,
        liquidity_backstop: LiquidityBackstopMode | None = None,
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
        self.config = _resolve_market_config(
            config,
            self.levels,
            preset=preset,
            logging_mode=logging_mode,
            liquidity_backstop=liquidity_backstop,
        )
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
        self.__engine = _MarketEngine(self)

        self._seed_initial_book(init_tick)
        self._record_current_state(features=self._compute_features())

    def _step_impl(self) -> None:
        self.__engine.run_step()

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

        return self.get_snapshot().to_dict()

    def get_snapshot(self) -> MarketSnapshot:
        """Return the current market snapshot as a typed view."""

        return _snapshot_from_mapping(self._history.current())

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

    def get_labeled_event_history(self) -> pd.DataFrame:
        """Return event history enriched with aligned latent debug labels."""

        self._require_full_logging("event/debug logging is disabled; use logging_mode='full'")
        return _merge_event_and_debug_history(self.get_event_history(), self.get_debug_history())

    def run(self, steps: int) -> SimulationResult:
        """Advance the simulator and return a bundled result view."""

        self.gen(steps)
        return self._build_result()

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

    def _build_result(self) -> SimulationResult:
        snapshot = self.get_snapshot()
        history = self.get_history()
        if self.config.logging_mode != "full":
            return SimulationResult(
                snapshot=snapshot,
                history=history,
                event_history=None,
                debug_history=None,
                labeled_event_history=None,
            )

        event_history = self.get_event_history()
        debug_history = self.get_debug_history()
        return SimulationResult(
            snapshot=snapshot,
            history=history,
            event_history=event_history,
            debug_history=debug_history,
            labeled_event_history=_merge_event_and_debug_history(event_history, debug_history),
        )


def _resolve_market_config(
    config: MarketConfig | Mapping[str, object] | None,
    levels: int,
    *,
    preset: PresetName | None,
    logging_mode: LoggingMode | None,
    liquidity_backstop: LiquidityBackstopMode | None,
) -> MarketConfig:
    overrides = {
        key: value
        for key, value in (
            ("preset", preset),
            ("logging_mode", logging_mode),
            ("liquidity_backstop", liquidity_backstop),
        )
        if value is not None
    }
    if not overrides:
        return coerce_config(config, levels)

    if config is None:
        return coerce_config(overrides, levels)
    if isinstance(config, MarketConfig):
        return coerce_config(replace(config, **overrides), levels)
    if isinstance(config, Mapping):
        merged = dict(config)
        merged.update(overrides)
        return coerce_config(merged, levels)
    return coerce_config(config, levels)


def _snapshot_from_mapping(snapshot: Mapping[str, object]) -> MarketSnapshot:
    bid_levels = cast(Sequence[object], snapshot["bids"])
    ask_levels = cast(Sequence[object], snapshot["asks"])
    return MarketSnapshot(
        step=cast(int, snapshot["step"]),
        day=cast(int, snapshot["day"]),
        session_step=cast(int, snapshot["session_step"]),
        session_phase=str(snapshot["session_phase"]),
        last_price=cast(float, snapshot["last_price"]),
        mid_price=cast(float, snapshot["mid_price"]),
        microprice=cast(float, snapshot["microprice"]),
        best_bid=cast(float, snapshot["best_bid"]),
        best_ask=cast(float, snapshot["best_ask"]),
        spread=cast(float, snapshot["spread"]),
        bids=tuple(_book_level_from_mapping(level) for level in bid_levels),
        asks=tuple(_book_level_from_mapping(level) for level in ask_levels),
        last_trade_side=None if snapshot["last_trade_side"] is None else str(snapshot["last_trade_side"]),
        last_trade_qty=cast(float, snapshot["last_trade_qty"]),
        buy_aggr_volume=cast(float, snapshot["buy_aggr_volume"]),
        sell_aggr_volume=cast(float, snapshot["sell_aggr_volume"]),
        trade_strength=cast(float, snapshot["trade_strength"]),
        depth_imbalance=cast(float, snapshot["depth_imbalance"]),
        regime=cast(RegimeName, snapshot["regime"]),
    )


def _book_level_from_mapping(level: object) -> BookLevel:
    if not isinstance(level, Mapping):
        raise TypeError("snapshot book levels must be mappings")
    return BookLevel(price=float(level["price"]), qty=float(level["qty"]))


def _merge_event_and_debug_history(event_history: pd.DataFrame, debug_history: pd.DataFrame) -> pd.DataFrame:
    debug_without_duplicate_context = debug_history.drop(
        columns=["day", "session_step", "session_phase"],
        errors="ignore",
    )
    return event_history.merge(
        debug_without_duplicate_context,
        on=["step", "event_idx"],
        how="inner",
        validate="one_to_one",
    )


__all__ = ["BookLevel", "Market", "MarketSnapshot", "SimulationResult"]
