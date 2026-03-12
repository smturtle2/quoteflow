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
_LimitFamily = Literal["join", "refill", "gap", "deep"]


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
class _KernelState:
    buy_flow_impulse: float = 0.0
    sell_flow_impulse: float = 0.0
    bid_cancel_pressure: float = 0.0
    ask_cancel_pressure: float = 0.0
    bid_refill_lag: float = 0.0
    ask_refill_lag: float = 0.0
    gap_pressure: float = 0.0


@dataclass(frozen=True)
class _KernelParams:
    flow_decay: float = 0.94
    cancel_decay: float = 0.90
    refill_decay: float = 0.88
    gap_decay: float = 0.86
    flow_to_fair: float = 0.24
    fair_to_pressure: float = 0.52
    imbalance_to_pressure: float = 0.18
    flow_to_pressure: float = 0.30
    gap_to_pressure: float = 0.08
    market_pressure_scale: float = 0.70
    market_flow_scale: float = 0.85
    market_cancel_scale: float = 0.24
    limit_pressure_scale: float = 0.42
    limit_gap_scale: float = 0.30
    limit_cancel_scale: float = 0.28
    limit_lag_scale: float = 0.48
    cancel_pressure_scale: float = 0.86
    cancel_gap_scale: float = 0.18
    cancel_signal_scale: float = 0.18
    taker_to_flow: float = 0.66
    taker_to_cancel: float = 0.78
    taker_to_gap: float = 0.48
    taker_to_refill_lag: float = 0.54
    cancel_to_gap: float = 0.28
    cancel_to_refill_lag: float = 0.18
    refill_relief: float = 0.70
    gap_fill_relief: float = 0.60
    deep_relief: float = 0.14
    side_cross_relief: float = 0.08
    join_size_scale: float = 0.62
    refill_size_scale: float = 1.02
    gap_size_scale: float = 0.94
    deep_size_scale: float = 1.28
    cancel_size_scale: float = 0.54
    market_size_scale: float = 0.78
    emergency_compress_scale: float = 0.88


@dataclass(frozen=True)
class _EventPlan:
    bid_cancel: int
    ask_cancel: int
    buy_market: int
    sell_market: int
    bid_join: int
    ask_join: int
    bid_refill: int
    ask_refill: int
    bid_gap_fill: int
    ask_gap_fill: int
    bid_deep: int
    ask_deep: int


class Market:
    """Compact aggregate order-book simulator with a queue-reactive internal kernel."""

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
        self._kernel_params = _KernelParams()
        self._rng = np.random.default_rng(seed)

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer()
        self._step = 0
        self._book_capacity = self.levels + self.config.max_spread_ticks + 4
        self._visual_window_ticks = max(
            12,
            self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks + 2,
        )
        self._visual_history = (
            VisualHistoryStore(depth_window_ticks=self._visual_window_ticks) if self.capture == "visual" else None
        )
        self._summary_state_cache: _SummaryState | None = None

        self._init_tick = price_to_tick(init_price, self.tick_size)
        self._last_trade_tick = self._init_tick
        self._fair_tick = float(self._init_tick)
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0
        self._kernel = _KernelState()

        self._seed_book(self._init_tick)
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

        self._decay_kernel_state()
        self._advance_fair_price()
        state = self._event_state()
        plan = self._sample_event_plan(state)

        for _ in range(plan.bid_cancel):
            self._apply_cancel_event("bid")
        for _ in range(plan.ask_cancel):
            self._apply_cancel_event("ask")

        for aggressor in self._market_sequence(plan.buy_market, plan.sell_market):
            self._apply_market_slice(aggressor)

        for side, family, count in self._limit_sequence(plan):
            for _ in range(count):
                self._apply_limit_event(side, family)

        self._repair_book()
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
            center_tick=int(round(mid_tick)),
        )
        return self._summary_state_cache

    def _event_state(self) -> _SummaryState:
        best_bid_tick = self._book.best_bid_tick
        best_ask_tick = self._book.best_ask_tick
        reference_tick = int(round(self._reference_tick()))
        if best_bid_tick is None:
            synthetic_bid = reference_tick - 1
            if best_ask_tick is not None:
                synthetic_bid = min(synthetic_bid, best_ask_tick - 1)
            best_bid_tick = synthetic_bid
        if best_ask_tick is None:
            synthetic_ask = reference_tick + 1
            synthetic_ask = max(synthetic_ask, best_bid_tick + 1)
            best_ask_tick = synthetic_ask
        if best_bid_tick >= best_ask_tick:
            best_bid_tick = min(best_bid_tick, best_ask_tick - 1)
            best_ask_tick = max(best_ask_tick, best_bid_tick + 1)
        mid_tick = (best_bid_tick + best_ask_tick) / 2.0
        spread_ticks = best_ask_tick - best_bid_tick
        bid_depth = self._book.total_depth("bid", self.levels)
        ask_depth = self._book.total_depth("ask", self.levels)
        return _SummaryState(
            best_bid_tick=best_bid_tick,
            best_ask_tick=best_ask_tick,
            mid_tick=mid_tick,
            spread_ticks=spread_ticks,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_imbalance=compute_depth_imbalance(bid_depth, ask_depth),
            center_tick=int(round(mid_tick)),
        )

    def _invalidate_summary_state(self) -> None:
        self._summary_state_cache = None

    def _seed_book(self, center_tick: int) -> None:
        for offset in range(self._book_capacity):
            size_scale = 0.90 + (0.12 * min(offset, 5))
            self._book.add_limit("bid", center_tick - 1 - offset, self._scaled_quantity(size_scale))
            self._book.add_limit("ask", center_tick + 1 + offset, self._scaled_quantity(size_scale))
        self._invalidate_summary_state()

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

    def _coerce_steps(self, steps: int) -> int:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return int(steps)

    def _decay_kernel_state(self) -> None:
        params = self._kernel_params
        self._kernel.buy_flow_impulse *= params.flow_decay
        self._kernel.sell_flow_impulse *= params.flow_decay
        self._kernel.bid_cancel_pressure *= params.cancel_decay
        self._kernel.ask_cancel_pressure *= params.cancel_decay
        self._kernel.bid_refill_lag *= params.refill_decay
        self._kernel.ask_refill_lag *= params.refill_decay
        self._kernel.gap_pressure *= params.gap_decay

    def _advance_fair_price(self) -> None:
        params = self._kernel_params
        reference_tick = self._reference_tick()
        net_flow = self._net_flow()
        reversion = self.config.mean_reversion * (reference_tick - self._fair_tick)
        shock = float(self._rng.normal(0.0, self.config.fair_price_vol))
        move = clamp(
            reversion + shock + (params.flow_to_fair * net_flow),
            -float(self.config.max_fair_move_ticks),
            float(self.config.max_fair_move_ticks),
        )
        self._fair_tick += move

    def _sample_event_plan(self, state: _SummaryState) -> _EventPlan:
        bid_limit = self._poisson_count(self._limit_rate("bid", state))
        ask_limit = self._poisson_count(self._limit_rate("ask", state))
        bid_cancel = self._poisson_count(self._cancel_rate("bid", state))
        ask_cancel = self._poisson_count(self._cancel_rate("ask", state))
        buy_market = self._poisson_count(self._market_rate("buy", state))
        sell_market = self._poisson_count(self._market_rate("sell", state))
        bid_join, bid_refill, bid_gap, bid_deep = self._split_limit_counts("bid", bid_limit, state)
        ask_join, ask_refill, ask_gap, ask_deep = self._split_limit_counts("ask", ask_limit, state)
        return _EventPlan(
            bid_cancel=bid_cancel,
            ask_cancel=ask_cancel,
            buy_market=buy_market,
            sell_market=sell_market,
            bid_join=bid_join,
            ask_join=ask_join,
            bid_refill=bid_refill,
            ask_refill=ask_refill,
            bid_gap_fill=bid_gap,
            ask_gap_fill=ask_gap,
            bid_deep=bid_deep,
            ask_deep=ask_deep,
        )

    def _poisson_count(self, rate: float) -> int:
        return int(self._rng.poisson(max(rate, 0.0)))

    def _split_limit_counts(self, side: BookSide, count: int, state: _SummaryState) -> tuple[int, int, int, int]:
        if count <= 0:
            return (0, 0, 0, 0)
        signal = self._side_signal(side, state)
        own_cancel = self._side_cancel_pressure(side)
        own_refill_lag = self._side_refill_lag(side)
        gap_flag = 1.0 if self._open_gap_ticks(side, max_pairs=6) else 0.0
        spread_component = self._spread_component(state)

        join_weight = max(
            0.05,
            0.16 + (0.68 * max(0.0, signal)) + (0.55 * spread_component) - (0.42 * own_refill_lag),
        )
        refill_weight = max(
            0.05,
            0.30 + (0.50 * own_cancel) + (0.36 * spread_component) - (0.22 * own_refill_lag),
        )
        gap_weight = max(
            0.05,
            0.18 + (0.90 * gap_flag) + (0.45 * self._kernel.gap_pressure) + (0.18 * own_cancel),
        )
        deep_weight = max(
            0.05,
            0.22 + (0.35 * max(0.0, -signal)) + (0.40 * own_refill_lag) + (0.16 * self._kernel.gap_pressure),
        )

        weights = np.array([join_weight, refill_weight, gap_weight, deep_weight], dtype=float)
        split = self._rng.multinomial(count, weights / weights.sum())
        return int(split[0]), int(split[1]), int(split[2]), int(split[3])

    def _market_rate(self, aggressor: AggressorSide, state: _SummaryState) -> float:
        params = self._kernel_params
        signal = self._side_signal("bid" if aggressor == "buy" else "ask", state)
        flow_impulse = self._kernel.buy_flow_impulse if aggressor == "buy" else self._kernel.sell_flow_impulse
        opposite_cancel = self._kernel.ask_cancel_pressure if aggressor == "buy" else self._kernel.bid_cancel_pressure
        spread_penalty = self._spread_component(state)
        base_rate = self.config.market_rate * 0.52
        scale = (
            0.62
            + (params.market_pressure_scale * max(0.0, signal))
            + (params.market_flow_scale * flow_impulse)
            + (params.market_cancel_scale * opposite_cancel)
            + (0.10 * self._kernel.gap_pressure)
            - (0.22 * spread_penalty)
        )
        return base_rate * max(0.08, scale)

    def _limit_rate(self, side: BookSide, state: _SummaryState) -> float:
        params = self._kernel_params
        signal = self._side_signal(side, state)
        own_cancel = self._side_cancel_pressure(side)
        own_refill_lag = self._side_refill_lag(side)
        base_rate = self.config.limit_rate * 0.50
        scale = (
            0.78
            + (params.limit_pressure_scale * max(0.0, signal))
            + (params.limit_gap_scale * self._kernel.gap_pressure)
            + (params.limit_cancel_scale * own_cancel)
            - (params.limit_lag_scale * own_refill_lag)
            + (0.32 * self._spread_component(state))
        )
        return base_rate * max(0.10, scale)

    def _cancel_rate(self, side: BookSide, state: _SummaryState) -> float:
        params = self._kernel_params
        signal = abs(self._side_signal(side, state))
        base_rate = self.config.cancel_rate * 0.50
        scale = (
            0.66
            + (params.cancel_pressure_scale * self._side_cancel_pressure(side))
            + (params.cancel_gap_scale * self._kernel.gap_pressure)
            + (params.cancel_signal_scale * signal)
        )
        return base_rate * max(0.10, scale)

    def _pressure_signal(self, state: _SummaryState) -> float:
        params = self._kernel_params
        fair_signal = clamp(
            (self._fair_tick - state.mid_tick) / float(self.config.max_fair_move_ticks),
            -1.0,
            1.0,
        )
        return clamp(
            (params.fair_to_pressure * fair_signal)
            + (params.imbalance_to_pressure * state.depth_imbalance)
            + (params.flow_to_pressure * self._net_flow())
            - (params.gap_to_pressure * self._kernel.gap_pressure),
            -1.0,
            1.0,
        )

    def _side_signal(self, side: BookSide, state: _SummaryState) -> float:
        pressure = self._pressure_signal(state)
        return pressure if side == "bid" else -pressure

    def _net_flow(self) -> float:
        return self._kernel.buy_flow_impulse - self._kernel.sell_flow_impulse

    def _spread_component(self, state: _SummaryState) -> float:
        if self.config.max_spread_ticks <= 1:
            return 0.0
        return clamp(
            (state.spread_ticks - 1) / float(self.config.max_spread_ticks - 1),
            0.0,
            1.0,
        )

    def _side_cancel_pressure(self, side: BookSide) -> float:
        return self._kernel.bid_cancel_pressure if side == "bid" else self._kernel.ask_cancel_pressure

    def _side_refill_lag(self, side: BookSide) -> float:
        return self._kernel.bid_refill_lag if side == "bid" else self._kernel.ask_refill_lag

    def _market_sequence(self, buy_count: int, sell_count: int) -> list[AggressorSide]:
        sequence: list[AggressorSide] = []
        sequence.extend(["buy"] * buy_count)
        sequence.extend(["sell"] * sell_count)
        if not sequence:
            return sequence
        favored_buy = self._kernel.buy_flow_impulse >= self._kernel.sell_flow_impulse
        if (buy_count > 0 and sell_count > 0) or not favored_buy:
            self._rng.shuffle(sequence)
        return sequence

    def _limit_sequence(self, plan: _EventPlan) -> tuple[tuple[BookSide, _LimitFamily, int], ...]:
        return (
            ("bid", "join", plan.bid_join),
            ("ask", "join", plan.ask_join),
            ("bid", "refill", plan.bid_refill),
            ("ask", "refill", plan.ask_refill),
            ("bid", "gap", plan.bid_gap_fill),
            ("ask", "gap", plan.ask_gap_fill),
            ("bid", "deep", plan.bid_deep),
            ("ask", "deep", plan.ask_deep),
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

    def _apply_market_slice(self, aggressor: AggressorSide) -> None:
        state = self._event_state()
        depth_before = max(1, state.ask_depth if aggressor == "buy" else state.bid_depth)
        qty_scale = self._kernel_params.market_size_scale
        flow_impulse = self._kernel.buy_flow_impulse if aggressor == "buy" else self._kernel.sell_flow_impulse
        qty = self._scaled_quantity(qty_scale + (0.22 * flow_impulse) + (0.18 * max(0.0, self._side_signal("bid" if aggressor == "buy" else "ask", state))))
        result = self._book.execute_market(aggressor, qty)
        if result.filled_qty <= 0:
            return

        if result.last_fill_tick is not None:
            self._last_trade_tick = result.last_fill_tick
        filled_qty = float(result.filled_qty)
        impact = clamp(filled_qty / float(depth_before), 0.0, 1.8)

        if aggressor == "buy":
            self._buy_aggr_volume += filled_qty
            self._on_market_pressure("buy", impact)
        else:
            self._sell_aggr_volume += filled_qty
            self._on_market_pressure("sell", impact)
        self._invalidate_summary_state()

    def _on_market_pressure(self, aggressor: AggressorSide, impact: float) -> None:
        params = self._kernel_params
        if aggressor == "buy":
            self._kernel.buy_flow_impulse = clamp(
                (params.flow_decay * self._kernel.buy_flow_impulse) + (params.taker_to_flow * impact),
                0.0,
                3.0,
            )
            self._kernel.sell_flow_impulse *= 0.94
            self._kernel.ask_cancel_pressure = clamp(
                (self._kernel.ask_cancel_pressure * 0.82) + (params.taker_to_cancel * impact),
                0.0,
                3.0,
            )
            self._kernel.ask_refill_lag = clamp(
                max(self._kernel.ask_refill_lag * 0.86, params.taker_to_refill_lag * impact),
                0.0,
                2.5,
            )
            self._kernel.bid_cancel_pressure *= 0.94
            self._kernel.bid_refill_lag *= 0.94
        else:
            self._kernel.sell_flow_impulse = clamp(
                (params.flow_decay * self._kernel.sell_flow_impulse) + (params.taker_to_flow * impact),
                0.0,
                3.0,
            )
            self._kernel.buy_flow_impulse *= 0.94
            self._kernel.bid_cancel_pressure = clamp(
                (self._kernel.bid_cancel_pressure * 0.82) + (params.taker_to_cancel * impact),
                0.0,
                3.0,
            )
            self._kernel.bid_refill_lag = clamp(
                max(self._kernel.bid_refill_lag * 0.86, params.taker_to_refill_lag * impact),
                0.0,
                2.5,
            )
            self._kernel.ask_cancel_pressure *= 0.94
            self._kernel.ask_refill_lag *= 0.94
        self._kernel.gap_pressure = clamp(
            (self._kernel.gap_pressure * 0.84) + (params.taker_to_gap * impact),
            0.0,
            3.0,
        )

    def _apply_cancel_event(self, side: BookSide) -> None:
        tick = self._cancel_tick(side)
        if tick is None:
            return
        level_qty = self._book.level_qty(side, tick)
        if level_qty <= 0:
            return
        qty = min(level_qty, self._cancel_quantity(side))
        if qty <= 0:
            return
        canceled = self._book.cancel_level(side, tick, qty)
        if canceled <= 0:
            return
        best_tick = self._book.best_bid_tick if side == "bid" else self._book.best_ask_tick
        was_touch = best_tick == tick
        self._on_cancel_pressure(side, canceled=float(canceled), was_touch=was_touch)
        self._invalidate_summary_state()

    def _cancel_tick(self, side: BookSide) -> int | None:
        levels = self._book.levels(side, min(self._book_capacity, 8))
        if not levels:
            return None
        max_qty = max(qty for _, qty in levels)
        weights: list[float] = []
        for index, (tick, qty) in enumerate(levels):
            thinness = 1.0 - (float(qty) / float(max_qty if max_qty > 0 else 1))
            gap_adjacent = 1.0 if self._tick_has_gap_neighbor(side, tick, levels) else 0.0
            weights.append(
                max(
                    0.05,
                    (1.90 / float(index + 1))
                    + (0.80 * thinness)
                    + (0.60 * gap_adjacent)
                    + (0.45 * self._side_cancel_pressure(side)),
                )
            )
        return levels[self._weighted_index(weights)][0]

    def _cancel_quantity(self, side: BookSide) -> int:
        scale = self._kernel_params.cancel_size_scale + (0.18 * self._side_cancel_pressure(side)) + (0.12 * self._kernel.gap_pressure)
        return self._scaled_quantity(scale)

    def _tick_has_gap_neighbor(self, side: BookSide, tick: int, levels: tuple[tuple[int, int], ...]) -> bool:
        level_ticks = [level_tick for level_tick, _ in levels]
        index = level_ticks.index(tick)
        if index > 0:
            upper = level_ticks[index - 1]
            if abs(upper - tick) > 1:
                return True
        if index + 1 < len(level_ticks):
            lower = level_ticks[index + 1]
            if abs(tick - lower) > 1:
                return True
        return False

    def _on_cancel_pressure(self, side: BookSide, *, canceled: float, was_touch: bool) -> None:
        params = self._kernel_params
        impact = clamp(canceled / float(max(self.config.max_order_qty, 1)), 0.0, 1.0)
        if side == "bid":
            self._kernel.bid_cancel_pressure = clamp(
                (self._kernel.bid_cancel_pressure * 0.88) + impact,
                0.0,
                3.0,
            )
            self._kernel.bid_refill_lag = clamp(
                max(self._kernel.bid_refill_lag * 0.92, params.cancel_to_refill_lag * impact),
                0.0,
                2.5,
            )
        else:
            self._kernel.ask_cancel_pressure = clamp(
                (self._kernel.ask_cancel_pressure * 0.88) + impact,
                0.0,
                3.0,
            )
            self._kernel.ask_refill_lag = clamp(
                max(self._kernel.ask_refill_lag * 0.92, params.cancel_to_refill_lag * impact),
                0.0,
                2.5,
            )
        if was_touch:
            self._kernel.gap_pressure = clamp(
                (self._kernel.gap_pressure * 0.88) + (params.cancel_to_gap * impact),
                0.0,
                3.0,
            )

    def _apply_limit_event(self, side: BookSide, family: _LimitFamily) -> None:
        state = self._event_state()
        signal = self._side_signal(side, state)
        if family == "join":
            tick = self._join_tick(side, state)
            qty = self._scaled_quantity(self._kernel_params.join_size_scale + (0.18 * max(0.0, signal)))
        elif family == "refill":
            tick = self._refill_tick(side, state)
            qty = self._scaled_quantity(
                self._kernel_params.refill_size_scale
                + (0.24 * self._side_cancel_pressure(side))
                - (0.14 * self._side_refill_lag(side))
            )
        elif family == "gap":
            tick = self._gap_fill_tick(side, state)
            qty = self._scaled_quantity(
                self._kernel_params.gap_size_scale
                + (0.28 * self._kernel.gap_pressure)
                + (0.10 * self._side_cancel_pressure(side))
            )
        else:
            tick = self._deep_add_tick(side, state)
            qty = self._scaled_quantity(
                self._kernel_params.deep_size_scale
                + (0.20 * self._side_refill_lag(side))
                + (0.12 * max(0.0, -signal))
            )
        self._book.add_limit(side, tick, qty)
        self._on_limit_relief(side, family)
        self._invalidate_summary_state()

    def _join_tick(self, side: BookSide, state: _SummaryState) -> int:
        if side == "bid":
            if state.spread_ticks > 1:
                return state.best_bid_tick + 1
            return state.best_bid_tick
        if state.spread_ticks > 1:
            return state.best_ask_tick - 1
        return state.best_ask_tick

    def _refill_tick(self, side: BookSide, state: _SummaryState) -> int:
        levels = self._book.levels(side, min(max(self.levels, 3), self._book_capacity))
        if not levels:
            return self._join_tick(side, state)
        top = levels[: min(3, len(levels))]
        max_qty = max(qty for _, qty in top)
        weights: list[float] = []
        for index, (_, qty) in enumerate(top):
            thinness = 1.0 - (float(qty) / float(max(max_qty, 1)))
            weights.append(max(0.05, (1.40 / float(index + 1)) + (0.85 * thinness)))
        return top[self._weighted_index(weights)][0]

    def _gap_fill_tick(self, side: BookSide, state: _SummaryState) -> int:
        candidates = self._open_gap_ticks(side, max_pairs=8)
        if not candidates:
            return self._deep_add_tick(side, state)
        weights = np.linspace(float(len(candidates)), 1.0, num=len(candidates), dtype=float) + (0.30 * self._kernel.gap_pressure)
        return candidates[self._weighted_index(weights)]

    def _deep_add_tick(self, side: BookSide, state: _SummaryState) -> int:
        candidates, weights = self._deep_candidates(side, state)
        if candidates:
            return candidates[self._weighted_index(weights)]
        return self._emergency_quote_tick(side, state)

    def _open_gap_ticks(self, side: BookSide, *, max_pairs: int) -> list[int]:
        levels = self._book.levels(side, min(self._book_capacity, max_pairs + 1))
        candidates: list[int] = []
        for index in range(len(levels) - 1):
            current_tick = levels[index][0]
            next_tick = levels[index + 1][0]
            gap = abs(current_tick - next_tick)
            if gap <= 1:
                continue
            if side == "bid":
                candidates.append(current_tick - 1)
            else:
                candidates.append(current_tick + 1)
        return candidates

    def _deep_candidates(self, side: BookSide, state: _SummaryState) -> tuple[list[int], list[float]]:
        occupied = {tick for tick, _ in self._book.levels(side)}
        candidates: list[int] = []
        weights: list[float] = []
        max_distance = self._book_capacity + self.config.max_spread_ticks + 2
        best_tick = state.best_bid_tick if side == "bid" else state.best_ask_tick
        for distance in range(2, max_distance + 1):
            tick = best_tick - distance if side == "bid" else best_tick + distance
            if side == "bid" and tick < 0:
                break
            if tick in occupied:
                continue
            if side == "bid" and tick >= state.best_ask_tick:
                continue
            if side == "ask" and tick <= state.best_bid_tick:
                continue
            neighbor_score = 1.0
            if side == "bid":
                if (tick + 1) in occupied:
                    neighbor_score += 0.35
                if (tick - 1) in occupied:
                    neighbor_score += 0.18
            else:
                if (tick - 1) in occupied:
                    neighbor_score += 0.35
                if (tick + 1) in occupied:
                    neighbor_score += 0.18
            distance_weight = float(self.config.level_decay ** max(distance - 2, 0))
            weights.append(max(0.02, distance_weight * neighbor_score))
            candidates.append(tick)
        return candidates, weights

    def _weighted_index(self, weights: list[float] | np.ndarray) -> int:
        normalized = np.asarray(weights, dtype=float)
        normalized = normalized / normalized.sum()
        return int(self._rng.choice(normalized.size, p=normalized))

    def _on_limit_relief(self, side: BookSide, family: _LimitFamily) -> None:
        params = self._kernel_params
        if side == "bid":
            self._kernel.bid_cancel_pressure *= 1.0 - params.side_cross_relief
            self._kernel.bid_refill_lag *= params.refill_relief
        else:
            self._kernel.ask_cancel_pressure *= 1.0 - params.side_cross_relief
            self._kernel.ask_refill_lag *= params.refill_relief
        if family == "gap":
            self._kernel.gap_pressure *= params.gap_fill_relief
        elif family == "deep":
            self._kernel.gap_pressure *= 1.0 - params.deep_relief

    def _repair_book(self) -> None:
        self._ensure_two_sided()
        self._book.clear_crossed_quotes()
        self._ensure_two_sided()
        self._compress_spread()
        self._book.trim("bid", self._book_capacity)
        self._book.trim("ask", self._book_capacity)
        self._ensure_two_sided()
        self._invalidate_summary_state()

    def _ensure_two_sided(self) -> None:
        state = self._event_state()
        if self._book.best_bid_tick is None:
            self._book.add_limit("bid", self._emergency_quote_tick("bid", state), self._scaled_quantity(0.90))
        if self._book.best_ask_tick is None:
            state = self._event_state()
            self._book.add_limit("ask", self._emergency_quote_tick("ask", state), self._scaled_quantity(0.90))

    def _compress_spread(self) -> None:
        state = self._event_state()
        if state.spread_ticks <= self.config.max_spread_ticks:
            return
        favored_bid = self._pressure_signal(state) >= 0.0
        if favored_bid:
            tick = max(0, state.best_ask_tick - self.config.max_spread_ticks)
            tick = min(tick, state.best_ask_tick - 1)
            self._book.add_limit("bid", tick, self._scaled_quantity(self._kernel_params.emergency_compress_scale))
        else:
            tick = state.best_bid_tick + self.config.max_spread_ticks
            tick = max(tick, state.best_bid_tick + 1)
            self._book.add_limit("ask", tick, self._scaled_quantity(self._kernel_params.emergency_compress_scale))

    def _emergency_quote_tick(self, side: BookSide, state: _SummaryState) -> int:
        reference_tick = int(round(self._reference_tick()))
        if side == "bid":
            return max(0, min(reference_tick - 1, state.best_ask_tick - 1))
        return max(reference_tick + 1, state.best_bid_tick + 1)

    def _public_levels(self, side: BookSide) -> list[dict[str, float]]:
        return [
            {"price": tick_to_price(tick, self.tick_size), "qty": float(qty)}
            for tick, qty in self._book.levels(side, self.levels)
        ]

    def _reference_tick(self) -> float:
        mid_tick = self._book.mid_tick()
        if mid_tick is not None:
            return mid_tick
        return float(self._last_trade_tick)

    def _require_visual_capture(self) -> VisualHistoryStore:
        if self._visual_history is None:
            raise RuntimeError("plot() and plot_heatmap() require Market(..., capture='visual')")
        return self._visual_history


__all__ = ["Market", "SimulationResult"]
