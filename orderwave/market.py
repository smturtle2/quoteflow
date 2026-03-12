from __future__ import annotations

"""Public market simulator API."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from orderwave.book import AggressorSide, BookSide, OrderBook
from orderwave.config import MarketConfig, coerce_config
from orderwave.history import HistoryBuffer
from orderwave.utils import bounded_int, clamp, compute_depth_imbalance, price_to_tick, tick_to_price
from orderwave.visualization import (
    VisualHistoryStore,
    plot_market_overview,
    plot_order_book,
)
from orderwave.visualization import (
    plot_heatmap as render_heatmap,
)

CaptureMode = Literal["summary", "visual"]


@dataclass(frozen=True)
class SimulationResult:
    """Collected outputs returned by ``Market.run``."""

    snapshot: dict[str, object]
    history: pd.DataFrame


@dataclass(frozen=True)
class _SummaryState:
    best_bid_tick: int
    best_ask_tick: int
    mid_tick: float
    spread_ticks: int
    bid_depth: int
    ask_depth: int
    depth_imbalance: float
    center_tick: int


@dataclass
class _LatentState:
    total_liquidity_log: float
    side_bias: float
    flow_bias: float
    latent_fair_offset: float
    realized_flow_memory: float
    bid_field: np.ndarray
    ask_field: np.ndarray


@dataclass(frozen=True)
class _LatentParams:
    liquidity_persistence: float = 0.95
    liquidity_noise: float = 0.12
    side_persistence: float = 0.92
    side_noise: float = 0.08
    flow_persistence: float = 0.95
    flow_noise: float = 0.10
    flow_feedback: float = 0.52
    fair_coupling: float = 0.06
    offset_persistence: float = 0.95
    offset_noise: float = 0.08
    field_persistence: float = 0.88
    field_diffusion: float = 0.24
    field_noise: float = 0.12
    field_tilt_scale: float = 0.08
    side_share_amplitude: float = 0.06
    budget_scale: float = 1.10
    shortage_budget_scale: float = 0.92
    connectivity_scale: float = 0.26
    queue_gap_scale: float = 0.68
    queue_penalty_scale: float = 0.08
    cox_shape: float = 2.8
    cancel_intercept: float = -2.65
    cancel_rate_scale: float = 1.25
    cancel_adverse_scale: float = 0.32
    cancel_support_scale: float = 0.42
    cancel_shortage_scale: float = 1.40
    cancel_depth_scale: float = 0.10
    cancel_queue_scale: float = 0.05
    market_base_scale: float = 0.85
    market_signal_scale: float = 1.10
    market_cox_shape: float = 2.6
    market_size_scale: float = 0.95
    market_size_flow_scale: float = 0.26
    memory_persistence: float = 0.88
    safety_qty_scale: float = 0.75


class Market:
    """Aggregate order-book simulator with latent-liquidity Cox dynamics."""

    def __init__(
        self,
        init_price: float = 100.0,
        tick_size: float = 0.01,
        levels: int = 5,
        seed: int | None = None,
        config: MarketConfig | Mapping[str, object] | None = None,
        *,
        capture: CaptureMode = "summary",
    ) -> None:
        if tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        if levels <= 0:
            raise ValueError("levels must be positive")
        if capture not in {"summary", "visual"}:
            raise ValueError("capture must be 'summary' or 'visual'")

        self.tick_size = float(tick_size)
        self.levels = int(levels)
        self.seed = seed
        self.capture = capture
        self.config = coerce_config(config)
        self._params = _LatentParams()
        self._rng = np.random.default_rng(seed)

        self._step = 0
        self._center_tick = price_to_tick(init_price, self.tick_size)
        self._last_trade_tick = self._center_tick
        self._fair_tick = float(self._center_tick)

        self._depth_cells = max(
            18,
            self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks + 18,
        )
        self._book_capacity = self._depth_cells
        self._grid: np.ndarray = np.arange(self._depth_cells, dtype=float)
        self._base_profile: np.ndarray = np.power(self.config.level_decay, self._grid)
        self._shape_tilt: np.ndarray = (1.0 / (1.0 + (0.35 * self._grid))) - 0.25
        self._support_weights: np.ndarray = np.exp(-0.60 * self._grid)
        self._profile_target: np.ndarray = 0.90 + (2.80 * self._base_profile)

        self._visible_bid: np.ndarray = np.zeros(self._depth_cells, dtype=np.int64)
        self._visible_ask: np.ndarray = np.zeros(self._depth_cells, dtype=np.int64)
        self._latent = self._initial_latent_state()

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer()
        self._summary_state_cache: _SummaryState | None = None
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0

        self._visual_window_ticks = max(
            12,
            self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks + 2,
        )
        self._visual_history = (
            VisualHistoryStore(depth_window_ticks=self._visual_window_ticks) if self.capture == "visual" else None
        )

        self._seed_visible_book()
        self._sync_book_from_visible()
        self._fair_tick = self._summary_state().mid_tick
        self._record_state()

    def get(self) -> dict[str, object]:
        """Return the current public market snapshot."""

        state = self._summary_state()
        return {
            "step": self._step,
            "last_price": tick_to_price(self._last_trade_tick, self.tick_size),
            "mid_price": tick_to_price(state.mid_tick, self.tick_size),
            "best_bid": tick_to_price(state.best_bid_tick, self.tick_size),
            "best_ask": tick_to_price(state.best_ask_tick, self.tick_size),
            "spread": tick_to_price(state.spread_ticks, self.tick_size),
            "bids": self._public_levels("bid"),
            "asks": self._public_levels("ask"),
            "bid_depth": state.bid_depth,
            "ask_depth": state.ask_depth,
            "depth_imbalance": state.depth_imbalance,
            "buy_aggr_volume": float(self._buy_aggr_volume),
            "sell_aggr_volume": float(self._sell_aggr_volume),
            "fair_price": tick_to_price(self._fair_tick, self.tick_size),
        }

    def get_history(self) -> pd.DataFrame:
        """Return the summary history as a ``pandas.DataFrame``."""

        return self._history.dataframe()

    def step(self) -> dict[str, object]:
        """Advance the simulator by one step and return the new snapshot."""

        self._step += 1
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0

        state = self._summary_state()
        fair_gap = self._advance_latent_state(state)
        bid_mean, ask_mean = self._latent_reveal_means(fair_gap, state.depth_imbalance)
        self._apply_cancel_thinning(fair_gap, bid_mean, ask_mean)
        self._apply_limit_reveals(bid_mean, ask_mean)
        self._apply_market_flow(fair_gap, bid_mean, ask_mean)
        self._repair_visible_arrays()
        self._recenter_visible_arrays()
        post_state = self._visible_state()
        post_fair_gap = self._fair_gap(post_state.mid_tick)
        post_bid_mean, post_ask_mean = self._latent_reveal_means(post_fair_gap, post_state.depth_imbalance)
        self._apply_limit_reveals(post_bid_mean, post_ask_mean, scale=0.65)
        self._repair_visible_arrays()
        self._sync_book_from_visible()
        self._record_state()
        return self.get()

    def gen(self, steps: int) -> dict[str, object]:
        """Run ``steps`` iterations and return the final snapshot."""

        for _ in range(self._coerce_steps(steps)):
            self.step()
        return self.get()

    def run(self, steps: int) -> SimulationResult:
        """Run ``steps`` iterations and return both snapshot and history."""

        snapshot = self.gen(steps)
        return SimulationResult(snapshot=snapshot, history=self.get_history())

    def plot(
        self,
        *,
        max_steps: int = 1200,
        price_window_ticks: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Render the overview plot with a level-ranked heatmap."""

        store = self._require_visual_capture()
        return plot_market_overview(
            self.get_history(),
            store,
            tick_size=self.tick_size,
            max_steps=max_steps,
            price_window_ticks=price_window_ticks,
            title=title,
            figsize=figsize,
        )

    def plot_heatmap(
        self,
        *,
        anchor: Literal["mid", "price"] = "mid",
        max_steps: int = 1200,
        price_window_ticks: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Render a standalone heatmap from captured visual history."""

        store = self._require_visual_capture()
        return render_heatmap(
            store,
            tick_size=self.tick_size,
            anchor=anchor,
            max_steps=max_steps,
            price_window_ticks=price_window_ticks,
            title=title,
            figsize=figsize,
        )

    def plot_book(
        self,
        *,
        levels: int | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Render the current order book."""

        return plot_order_book(
            self._book,
            tick_size=self.tick_size,
            levels=levels or self.levels,
            title=title,
            figsize=figsize,
        )

    def _initial_latent_state(self) -> _LatentState:
        common_field = 0.06 * self._rng.normal(size=self._depth_cells)
        bid_field = common_field + (0.02 * self._rng.normal(size=self._depth_cells))
        ask_field = common_field + (0.02 * self._rng.normal(size=self._depth_cells))
        return _LatentState(
            total_liquidity_log=0.0,
            side_bias=0.0,
            flow_bias=0.0,
            latent_fair_offset=0.0,
            realized_flow_memory=0.0,
            bid_field=bid_field.astype(float),
            ask_field=ask_field.astype(float),
        )

    def _seed_visible_book(self) -> None:
        bid_mean, ask_mean = self._latent_reveal_means(0.0, 0.0)
        self._visible_bid = np.asarray(self._rng.poisson(bid_mean * 1.20), dtype=np.int64)
        self._visible_ask = np.asarray(self._rng.poisson(ask_mean * 1.20), dtype=np.int64)
        self._repair_visible_arrays()
        self._recenter_visible_arrays()

    def _summary_state(self) -> _SummaryState:
        if self._summary_state_cache is not None:
            return self._summary_state_cache
        best_bid_tick = self._book.best_bid_tick
        best_ask_tick = self._book.best_ask_tick
        if best_bid_tick is None or best_ask_tick is None:
            raise RuntimeError("order book must remain two-sided")
        mid_tick = (best_bid_tick + best_ask_tick) / 2.0
        spread_ticks = best_ask_tick - best_bid_tick
        bid_depth = self._book.total_depth("bid", self.levels)
        ask_depth = self._book.total_depth("ask", self.levels)
        depth_imbalance = compute_depth_imbalance(bid_depth, ask_depth)
        self._summary_state_cache = _SummaryState(
            best_bid_tick=best_bid_tick,
            best_ask_tick=best_ask_tick,
            mid_tick=mid_tick,
            spread_ticks=spread_ticks,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_imbalance=depth_imbalance,
            center_tick=_center_from_quotes(best_bid_tick, best_ask_tick),
        )
        return self._summary_state_cache

    def _invalidate_summary_state(self) -> None:
        self._summary_state_cache = None

    def _record_state(self) -> None:
        state = self._summary_state()
        self._history.append(
            step=self._step,
            last_price=tick_to_price(self._last_trade_tick, self.tick_size),
            mid_price=tick_to_price(state.mid_tick, self.tick_size),
            best_bid=tick_to_price(state.best_bid_tick, self.tick_size),
            best_ask=tick_to_price(state.best_ask_tick, self.tick_size),
            spread=tick_to_price(state.spread_ticks, self.tick_size),
            bid_depth=state.bid_depth,
            ask_depth=state.ask_depth,
            depth_imbalance=state.depth_imbalance,
            buy_aggr_volume=self._buy_aggr_volume,
            sell_aggr_volume=self._sell_aggr_volume,
            fair_price=tick_to_price(self._fair_tick, self.tick_size),
        )
        if self._visual_history is not None:
            self._visual_history.append(
                step=self._step,
                center_tick=state.center_tick,
                best_bid_tick=state.best_bid_tick,
                best_ask_tick=state.best_ask_tick,
                signed_depth=self._book.signed_window(state.center_tick, self._visual_window_ticks),
            )

    def _visible_state(self) -> _SummaryState:
        bid_index = _first_positive_index(self._visible_bid)
        ask_index = _first_positive_index(self._visible_ask)
        if bid_index is None:
            bid_index = 0
        if ask_index is None:
            ask_index = 0
        best_bid_tick = self._tick_at_distance("bid", bid_index)
        best_ask_tick = self._tick_at_distance("ask", ask_index)
        mid_tick = (best_bid_tick + best_ask_tick) / 2.0
        bid_depth = int(np.sum(self._visible_bid[: self.levels]))
        ask_depth = int(np.sum(self._visible_ask[: self.levels]))
        return _SummaryState(
            best_bid_tick=best_bid_tick,
            best_ask_tick=best_ask_tick,
            mid_tick=mid_tick,
            spread_ticks=best_ask_tick - best_bid_tick,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_imbalance=compute_depth_imbalance(bid_depth, ask_depth),
            center_tick=_center_from_quotes(best_bid_tick, best_ask_tick),
        )

    def _coerce_steps(self, steps: int) -> int:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return int(steps)

    def _advance_latent_state(self, state: _SummaryState) -> float:
        params = self._params
        fair_gap = self._fair_gap(state.mid_tick)
        spread_component = self._spread_component(state.spread_ticks)
        bid_support = self._side_support(self._visible_bid)
        ask_support = self._side_support(self._visible_ask)
        support_imbalance = _continuous_imbalance(bid_support, ask_support)
        mean_support = 0.5 * (bid_support + ask_support)
        target_support = 1.15 + (0.22 * self.config.limit_rate)

        self._latent.total_liquidity_log = (
            (params.liquidity_persistence * self._latent.total_liquidity_log)
            + (0.18 * clamp((target_support - mean_support) / max(target_support, 1.0), -1.0, 1.0))
            - (0.04 * spread_component)
            + float(self._rng.normal(0.0, params.liquidity_noise))
        )
        self._latent.flow_bias = clamp(
            (params.flow_persistence * self._latent.flow_bias)
            + (params.flow_feedback * self._latent.realized_flow_memory)
            + (params.fair_coupling * fair_gap)
            + float(self._rng.normal(0.0, params.flow_noise)),
            -2.0,
            2.0,
        )
        self._latent.side_bias = clamp(
            (params.side_persistence * self._latent.side_bias)
            + (0.06 * fair_gap)
            + (0.08 * self._latent.flow_bias)
            + float(self._rng.normal(0.0, params.side_noise)),
            -1.25,
            1.25,
        )
        self._latent.latent_fair_offset = clamp(
            (params.offset_persistence * self._latent.latent_fair_offset)
            + (0.03 * self._latent.flow_bias)
            + float(self._rng.normal(0.0, params.offset_noise)),
            -float(self.config.max_fair_move_ticks) * 1.5,
            float(self.config.max_fair_move_ticks) * 1.5,
        )

        drift_signal = (0.12 * fair_gap) - (0.08 * support_imbalance)
        drift = params.field_tilt_scale * drift_signal * self._shape_tilt
        self._latent.bid_field = self._advance_field(self._latent.bid_field, drift)
        self._latent.ask_field = self._advance_field(self._latent.ask_field, drift)
        common_field = 0.5 * (self._latent.bid_field + self._latent.ask_field)
        self._latent.bid_field = (0.84 * self._latent.bid_field) + (0.16 * common_field)
        self._latent.ask_field = (0.84 * self._latent.ask_field) + (0.16 * common_field)

        fair_anchor = state.mid_tick + self._latent.latent_fair_offset
        reversion = self.config.mean_reversion * (fair_anchor - self._fair_tick)
        shock = float(self._rng.normal(0.0, self.config.fair_price_vol))
        move = clamp(
            reversion + (0.05 * self._latent.flow_bias) + shock,
            -float(self.config.max_fair_move_ticks),
            float(self.config.max_fair_move_ticks),
        )
        self._fair_tick += move
        return self._fair_gap(state.mid_tick)


    def _advance_field(self, field: np.ndarray, drift: np.ndarray) -> np.ndarray:
        params = self._params
        neighbor = np.empty_like(field)
        neighbor[0] = field[1]
        neighbor[-1] = field[-2]
        neighbor[1:-1] = 0.5 * (field[:-2] + field[2:])
        noise = self._rng.normal(0.0, params.field_noise, size=self._depth_cells)
        advanced = (
            params.field_persistence * field
            + params.field_diffusion * (neighbor - field)
            + drift
            + noise
        )
        return np.clip(advanced, -3.0, 3.0)

    def _side_support(self, visible: np.ndarray) -> float:
        horizon = min(self.levels + 3, self._depth_cells)
        support = self._support_weights[:horizon] * np.log1p(visible[:horizon].astype(float))
        return float(np.sum(support))

    def _side_shortages(self) -> tuple[float, float]:
        target_support = 1.05 + (0.18 * self.config.limit_rate)
        bid_support = self._side_support(self._visible_bid)
        ask_support = self._side_support(self._visible_ask)
        bid_support_shortage = clamp((target_support - bid_support) / max(target_support, 1.0), -0.55, 1.25)
        ask_support_shortage = clamp((target_support - ask_support) / max(target_support, 1.0), -0.55, 1.25)

        bid_occupancy = float(np.sum(1.0 - np.exp(-self._visible_bid.astype(float) / 2.5)))
        ask_occupancy = float(np.sum(1.0 - np.exp(-self._visible_ask.astype(float) / 2.5)))
        occupancy_imbalance = _continuous_imbalance(bid_occupancy, ask_occupancy)

        bid_shortage = clamp((0.78 * bid_support_shortage) - (0.42 * occupancy_imbalance), -0.65, 1.25)
        ask_shortage = clamp((0.78 * ask_support_shortage) + (0.42 * occupancy_imbalance), -0.65, 1.25)
        return float(bid_shortage), float(ask_shortage)

    def _neighbor_log_support(self, log_visible: np.ndarray) -> np.ndarray:
        neighbor = np.empty_like(log_visible)
        neighbor[0] = log_visible[1]
        neighbor[-1] = log_visible[-2]
        neighbor[1:-1] = 0.5 * (log_visible[:-2] + log_visible[2:])
        return neighbor

    def _latent_side_budget(self, base_budget: float, shortage: float, skew: float) -> float:
        multiplier = 1.0 + (self._params.shortage_budget_scale * shortage) + (self._params.side_share_amplitude * skew)
        return base_budget * clamp(multiplier, 0.70, 1.75)

    def _cell_scores(self, visible: np.ndarray, field: np.ndarray, shortage: float) -> np.ndarray:
        log_visible = np.log1p(visible.astype(float))
        gap_target = self._profile_target * (1.0 + (0.35 * max(shortage, 0.0)))
        gap = np.clip(np.log1p(gap_target) - log_visible, -1.5, 1.5)
        neighbor = self._neighbor_log_support(log_visible)
        score = (
            self._base_profile
            * np.exp(np.clip(field, -2.0, 2.0))
            * np.exp(
                (self._params.queue_gap_scale * gap)
                + (self._params.connectivity_scale * neighbor)
                - (self._params.queue_penalty_scale * log_visible)
            )
        )
        return np.maximum(score, 1e-9)

    def _latent_reveal_means(self, fair_gap: float, depth_imbalance: float) -> tuple[np.ndarray, np.ndarray]:
        params = self._params
        base_budget = (
            params.budget_scale
            * self.config.limit_rate
            * float(np.exp(np.clip(self._latent.total_liquidity_log, -0.9, 0.9)))
        )
        bid_shortage, ask_shortage = self._side_shortages()
        split_signal = clamp((0.12 * self._latent.side_bias) + (0.16 * fair_gap) - (0.35 * depth_imbalance), -2.0, 2.0)
        skew = np.tanh(0.5 * split_signal)

        bid_budget = self._latent_side_budget(base_budget, bid_shortage, float(skew))
        ask_budget = self._latent_side_budget(base_budget, ask_shortage, float(-skew))
        common_field = 0.5 * (self._latent.bid_field + self._latent.ask_field)
        bid_field = (0.75 * common_field) + (0.25 * self._latent.bid_field)
        ask_field = (0.75 * common_field) + (0.25 * self._latent.ask_field)
        bid_score = self._cell_scores(self._visible_bid, bid_field, bid_shortage)
        ask_score = self._cell_scores(self._visible_ask, ask_field, ask_shortage)
        bid_mean = bid_budget * (bid_score / np.sum(bid_score))
        ask_mean = ask_budget * (ask_score / np.sum(ask_score))

        bid_mean = bid_mean * self._rng.gamma(params.cox_shape, 1.0 / params.cox_shape, size=self._depth_cells)
        ask_mean = ask_mean * self._rng.gamma(params.cox_shape, 1.0 / params.cox_shape, size=self._depth_cells)
        return bid_mean.astype(float), ask_mean.astype(float)


    def _apply_cancel_thinning(self, fair_gap: float, bid_mean: np.ndarray, ask_mean: np.ndarray) -> None:
        bid_shortage, ask_shortage = self._side_shortages()
        cancel_norm = self.config.cancel_rate / float(self.config.cancel_rate + self.config.limit_rate)
        latent_imbalance = _continuous_imbalance(float(np.sum(bid_mean[:4])), float(np.sum(ask_mean[:4])))
        depth_term = self._grid / max(float(self._depth_cells - 1), 1.0)

        bid_adverse = clamp(
            -(0.45 * self._latent.flow_bias) - (0.25 * fair_gap) - (0.15 * latent_imbalance),
            -2.5,
            2.5,
        )
        ask_adverse = clamp(
            (0.45 * self._latent.flow_bias) + (0.25 * fair_gap) + (0.15 * latent_imbalance),
            -2.5,
            2.5,
        )

        bid_signal = (
            self._params.cancel_intercept
            + (self._params.cancel_rate_scale * cancel_norm)
            + (self._params.cancel_adverse_scale * bid_adverse)
            + (self._params.cancel_depth_scale * depth_term)
            + (self._params.cancel_queue_scale * np.log1p(self._visible_bid.astype(float)))
            - (self._params.cancel_support_scale * np.log1p(bid_mean + 1.0))
            - (self._params.cancel_shortage_scale * bid_shortage)
        )
        ask_signal = (
            self._params.cancel_intercept
            + (self._params.cancel_rate_scale * cancel_norm)
            + (self._params.cancel_adverse_scale * ask_adverse)
            + (self._params.cancel_depth_scale * depth_term)
            + (self._params.cancel_queue_scale * np.log1p(self._visible_ask.astype(float)))
            - (self._params.cancel_support_scale * np.log1p(ask_mean + 1.0))
            - (self._params.cancel_shortage_scale * ask_shortage)
        )

        bid_cancel = self._rng.binomial(self._visible_bid, _sigmoid(bid_signal))
        ask_cancel = self._rng.binomial(self._visible_ask, _sigmoid(ask_signal))
        self._visible_bid = np.maximum(0, self._visible_bid - bid_cancel)
        self._visible_ask = np.maximum(0, self._visible_ask - ask_cancel)


    def _apply_limit_reveals(self, bid_mean: np.ndarray, ask_mean: np.ndarray, *, scale: float = 1.0) -> None:
        self._visible_bid = self._visible_bid + np.asarray(
            self._rng.poisson(bid_mean * max(scale, 0.0)),
            dtype=np.int64,
        )
        self._visible_ask = self._visible_ask + np.asarray(
            self._rng.poisson(ask_mean * max(scale, 0.0)),
            dtype=np.int64,
        )

    def _apply_market_flow(self, fair_gap: float, bid_mean: np.ndarray, ask_mean: np.ndarray) -> None:
        visible_imbalance = _continuous_imbalance(self._side_support(self._visible_bid), self._side_support(self._visible_ask))
        latent_imbalance = _continuous_imbalance(
            float(np.sum(bid_mean[:4] * self._support_weights[:4])),
            float(np.sum(ask_mean[:4] * self._support_weights[:4])),
        )
        signal = clamp(
            (0.40 * fair_gap)
            + (0.95 * self._latent.flow_bias)
            + (0.30 * self._latent.realized_flow_memory)
            - (0.14 * visible_imbalance)
            - (0.10 * latent_imbalance),
            -2.5,
            2.5,
        )
        base_rate = self._params.market_base_scale * self.config.market_rate
        activity = base_rate * _softplus(0.65 + (0.35 * abs(signal)))
        activity *= (1.0 + (0.20 * abs(self._latent.realized_flow_memory)))
        activity *= float(self._rng.gamma(self._params.market_cox_shape, 1.0 / self._params.market_cox_shape))

        total_count = int(self._rng.poisson(max(activity, 0.0)))
        buy_share = float(_sigmoid((self._params.market_signal_scale + 0.30) * signal))
        buy_count = int(self._rng.binomial(total_count, buy_share))
        sell_count = max(0, total_count - buy_count)
        sequence: list[AggressorSide] = []
        if buy_count > 0 or sell_count > 0:
            buy_first = (buy_count > sell_count) or (buy_count == sell_count and self._rng.random() < buy_share)
            if buy_first:
                sequence.extend(["buy"] * buy_count)
                sequence.extend(["sell"] * sell_count)
            else:
                sequence.extend(["sell"] * sell_count)
                sequence.extend(["buy"] * buy_count)

        depth_before = max(
            1.0,
            float(np.sum(self._visible_bid[: self.levels])) + float(np.sum(self._visible_ask[: self.levels])),
        )
        scale = self._params.market_size_scale + (self._params.market_size_flow_scale * abs(signal))
        for aggressor in sequence:
            front_depth = self._side_support(self._visible_ask if aggressor == "buy" else self._visible_bid)
            qty = self._scaled_quantity(scale * _slice_liquidity_scale(front_depth, self.levels))
            filled = self._execute_market_order(aggressor, qty)
            if aggressor == "buy":
                self._buy_aggr_volume += float(filled)
            else:
                self._sell_aggr_volume += float(filled)

        net_flow = (self._buy_aggr_volume - self._sell_aggr_volume) / depth_before
        self._latent.realized_flow_memory = clamp(
            (self._params.memory_persistence * self._latent.realized_flow_memory) + (1.05 * net_flow),
            -2.5,
            2.5,
        )


    def _execute_market_order(self, aggressor: AggressorSide, qty: int) -> int:
        remaining = int(qty)
        filled = 0
        if remaining <= 0:
            return 0

        visible = self._visible_ask if aggressor == "buy" else self._visible_bid
        side: BookSide = "ask" if aggressor == "buy" else "bid"
        for distance, resting in enumerate(visible):
            if remaining <= 0:
                break
            if resting <= 0:
                continue
            take = min(int(resting), remaining)
            visible[distance] -= take
            remaining -= take
            filled += take
            self._last_trade_tick = self._tick_at_distance(side, distance)
        return filled

    def _repair_visible_arrays(self) -> None:
        self._ensure_side_presence("bid")
        self._ensure_side_presence("ask")
        self._enforce_spread_cap()

    def _ensure_side_presence(self, side: BookSide) -> None:
        visible = self._visible_bid if side == "bid" else self._visible_ask
        if np.any(visible > 0):
            return
        visible[0] = max(visible[0], self._scaled_quantity(self._params.safety_qty_scale))

    def _enforce_spread_cap(self) -> None:
        for _ in range(self.config.max_spread_ticks + 2):
            bid_index = _first_positive_index(self._visible_bid)
            ask_index = _first_positive_index(self._visible_ask)
            if bid_index is None or ask_index is None:
                return
            spread_ticks = bid_index + ask_index + 1
            if spread_ticks <= self.config.max_spread_ticks:
                return

            target_bid = max(0, self.config.max_spread_ticks - ask_index - 1)
            target_ask = max(0, self.config.max_spread_ticks - bid_index - 1)
            if ask_index > bid_index:
                self._visible_ask[target_ask] += self._scaled_quantity(self._params.safety_qty_scale)
            elif bid_index > ask_index:
                self._visible_bid[target_bid] += self._scaled_quantity(self._params.safety_qty_scale)
            else:
                if self._fair_tick >= float(self._center_tick):
                    self._visible_bid[target_bid] += self._scaled_quantity(self._params.safety_qty_scale)
                else:
                    self._visible_ask[target_ask] += self._scaled_quantity(self._params.safety_qty_scale)

    def _recenter_visible_arrays(self) -> None:
        bid_index = _first_positive_index(self._visible_bid)
        ask_index = _first_positive_index(self._visible_ask)
        if bid_index is None or ask_index is None:
            return
        best_bid_tick = self._tick_at_distance("bid", bid_index)
        best_ask_tick = self._tick_at_distance("ask", ask_index)
        new_center = _center_from_quotes(best_bid_tick, best_ask_tick)
        delta = new_center - self._center_tick
        if delta == 0:
            return
        self._visible_bid = _shift_bid(self._visible_bid, delta)
        self._visible_ask = _shift_ask(self._visible_ask, delta)
        self._center_tick = new_center

    def _sync_book_from_visible(self) -> None:
        book = OrderBook(self.tick_size)
        for distance, qty in enumerate(self._visible_bid):
            if qty <= 0:
                continue
            book.add_limit("bid", self._tick_at_distance("bid", distance), int(qty))
        for distance, qty in enumerate(self._visible_ask):
            if qty <= 0:
                continue
            book.add_limit("ask", self._tick_at_distance("ask", distance), int(qty))
        self._book = book
        self._invalidate_summary_state()

    def _tick_at_distance(self, side: BookSide, distance: int) -> int:
        if side == "bid":
            return max(0, self._center_tick - int(distance))
        return self._center_tick + 1 + int(distance)

    def _spread_component(self, spread_ticks: int) -> float:
        if self.config.max_spread_ticks <= 1:
            return 0.0
        return clamp(
            (spread_ticks - 1) / float(self.config.max_spread_ticks - 1),
            0.0,
            1.5,
        )

    def _fair_gap(self, mid_tick: float) -> float:
        return clamp(
            (self._fair_tick - mid_tick) / float(self.config.max_fair_move_ticks),
            -2.0,
            2.0,
        )

    def _sample_quantity(self) -> int:
        raw_value = float(self._rng.lognormal(self.config.size_mean, self.config.size_dispersion))
        return bounded_int(raw_value, self.config.min_order_qty, self.config.max_order_qty)

    def _scaled_quantity(self, scale: float) -> int:
        return bounded_int(
            round(self._sample_quantity() * max(scale, 0.20)),
            self.config.min_order_qty,
            self.config.max_order_qty,
        )

    def _public_levels(self, side: BookSide) -> list[dict[str, float]]:
        return [
            {"price": tick_to_price(tick, self.tick_size), "qty": float(qty)}
            for tick, qty in self._book.levels(side, self.levels)
        ]

    def _require_visual_capture(self) -> VisualHistoryStore:
        if self._visual_history is None:
            raise RuntimeError("plot() and plot_heatmap() require Market(..., capture='visual')")
        return self._visual_history


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softplus(value: float) -> float:
    clipped = clamp(value, -40.0, 40.0)
    return float(np.log1p(np.exp(clipped)))


def _first_positive_index(values: np.ndarray) -> int | None:
    positions = np.flatnonzero(values > 0)
    if positions.size == 0:
        return None
    return int(positions[0])


def _center_from_quotes(best_bid_tick: int, best_ask_tick: int) -> int:
    spread_ticks = max(best_ask_tick - best_bid_tick, 1)
    return best_bid_tick + ((spread_ticks - 1) // 2)


def _shift_bid(values: np.ndarray, delta: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if delta >= 0:
        if delta < values.size:
            shifted[delta:] = values[: values.size - delta]
        return shifted
    offset = -delta
    if offset < values.size:
        shifted[: values.size - offset] = values[offset:]
    return shifted


def _shift_ask(values: np.ndarray, delta: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if delta >= 0:
        if delta < values.size:
            shifted[: values.size - delta] = values[delta:]
        return shifted
    offset = -delta
    if offset < values.size:
        shifted[offset:] = values[: values.size - offset]
    return shifted


def _support_scale(values: np.ndarray) -> np.ndarray:
    support = np.log1p(values.astype(float))
    smoothed = np.empty_like(support)
    smoothed[0] = support[0] + (0.55 * support[1])
    smoothed[-1] = support[-1] + (0.55 * support[-2])
    smoothed[1:-1] = (0.25 * support[:-2]) + support[1:-1] + (0.25 * support[2:])
    mean_value = float(np.mean(smoothed))
    if mean_value <= 1e-9:
        return np.ones_like(smoothed)
    normalized = smoothed / mean_value
    return 0.72 + (0.28 * np.clip(normalized, 0.35, 2.40))


def _slice_liquidity_scale(front_depth: float, levels: int) -> float:
    reference = max(float((2 * levels) + 2), 1.0)
    return 0.40 + (0.60 * np.tanh(max(front_depth, 0.0) / reference))


def _continuous_imbalance(lhs: float, rhs: float) -> float:
    total = lhs + rhs
    if total <= 1e-12:
        return 0.0
    return (lhs - rhs) / total


__all__ = ["Market", "SimulationResult"]
