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

_EVENT_LIMIT = 0
_EVENT_MARKET = 1
_EVENT_CANCEL = 2


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


class Market:
    """Compact aggregate order-book simulator driven by explicit distributions."""

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
        self._rng = np.random.default_rng(seed)

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer()
        self._step = 0
        self._book_capacity = self.levels + self.config.max_spread_ticks + 2
        self._visual_window_ticks = max(12, self.levels + self.config.max_spread_ticks + self.config.max_fair_move_ticks)
        self._visual_history = (
            VisualHistoryStore(depth_window_ticks=self._visual_window_ticks) if self.capture == "visual" else None
        )
        self._weight_cache: dict[int, np.ndarray] = {}
        self._summary_state_cache: _SummaryState | None = None

        self._init_tick = price_to_tick(init_price, self.tick_size)
        self._last_trade_tick = self._init_tick
        self._fair_tick = float(self._init_tick)
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0

        self._flow_ema = 0.0
        self._bid_stress = 0.0
        self._ask_stress = 0.0
        self._refill_pressure = 0.0

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
        event_codes = self._sample_event_codes()
        self._rng.shuffle(event_codes)
        for event_code in event_codes:
            if int(event_code) == _EVENT_LIMIT:
                self._apply_limit_event()
            elif int(event_code) == _EVENT_MARKET:
                self._apply_market_event()
            else:
                self._apply_cancel_event()

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
        """Render the overview plot with a mid-anchored heatmap."""

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

    def _invalidate_summary_state(self) -> None:
        self._summary_state_cache = None

    def _seed_book(self, center_tick: int) -> None:
        for offset in range(self._book_capacity):
            self._book.add_limit("bid", center_tick - 1 - offset, self._sample_quantity())
            self._book.add_limit("ask", center_tick + 1 + offset, self._sample_quantity())
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
        self._flow_ema *= 0.88
        self._bid_stress *= 0.84
        self._ask_stress *= 0.84
        self._refill_pressure *= 0.78

    def _advance_fair_price(self) -> None:
        reference_tick = self._reference_tick()
        reversion = self.config.mean_reversion * (reference_tick - self._fair_tick)
        shock = float(self._rng.normal(0.0, self.config.fair_price_vol))
        move = clamp(
            reversion + shock + (0.20 * self._flow_ema),
            -float(self.config.max_fair_move_ticks),
            float(self.config.max_fair_move_ticks),
        )
        self._fair_tick += move

    def _sample_event_codes(self):
        limit_count = int(self._rng.poisson(self.config.limit_rate))
        market_count = int(self._rng.poisson(self.config.market_rate))
        cancel_count = int(self._rng.poisson(self.config.cancel_rate))
        total = limit_count + market_count + cancel_count
        event_codes: np.ndarray = np.empty(total, dtype=np.int8)
        start = 0
        event_codes[start : start + limit_count] = _EVENT_LIMIT
        start += limit_count
        event_codes[start : start + market_count] = _EVENT_MARKET
        start += market_count
        event_codes[start : start + cancel_count] = _EVENT_CANCEL
        return event_codes

    def _sample_quantity(self) -> int:
        raw_value = float(self._rng.lognormal(self.config.size_mean, self.config.size_dispersion))
        return bounded_int(raw_value, self.config.min_order_qty, self.config.max_order_qty)

    def _scaled_quantity(self, scale: float) -> int:
        return bounded_int(
            round(self._sample_quantity() * max(scale, 0.25)),
            self.config.min_order_qty,
            self.config.max_order_qty,
        )

    def _pressure_signal(self, state: _SummaryState) -> float:
        fair_signal = clamp(
            (self._fair_tick - state.mid_tick) / float(self.config.max_fair_move_ticks),
            -1.0,
            1.0,
        )
        return clamp((0.55 * fair_signal) - (0.35 * state.depth_imbalance) + (0.30 * self._flow_ema), -1.0, 1.0)

    def _bid_probability(self, state: _SummaryState) -> float:
        return 0.5 * (1.0 + self._pressure_signal(state))

    def _apply_limit_event(self) -> None:
        state = self._summary_state()
        pressure = self._pressure_signal(state)
        side: BookSide = "bid" if self._rng.random() < self._bid_probability(state) else "ask"
        favored_pressure = pressure if side == "bid" else -pressure
        side_stress = self._bid_stress if side == "bid" else self._ask_stress
        opposite_stress = self._ask_stress if side == "bid" else self._bid_stress
        spread_component = 0.0
        if self.config.max_spread_ticks > 1:
            spread_component = clamp(
                (state.spread_ticks - 1) / float(self.config.max_spread_ticks - 1),
                0.0,
                1.0,
            )

        inside_weight = max(0.0, 0.15 + (0.75 * max(0.0, favored_pressure)) + (0.50 * spread_component) + (0.35 * opposite_stress))
        refill_weight = max(0.0, 0.40 + (0.50 * self._refill_pressure) + (0.45 * side_stress))
        wall_weight = max(0.0, 0.25 + (0.55 * max(0.0, -favored_pressure)) + (0.20 * state.spread_ticks))
        total = inside_weight + refill_weight + wall_weight
        draw = self._rng.random() * total

        tick: int
        if draw < inside_weight:
            tick = self._inside_tick(side)
            qty = self._scaled_quantity(1.00 + (0.30 * max(0.0, favored_pressure)))
        elif draw < inside_weight + refill_weight:
            best_tick = self._best_tick(side)
            tick = best_tick if best_tick is not None else self._inside_tick(side)
            qty = self._scaled_quantity(1.00 + (0.70 * self._refill_pressure) + (0.25 * side_stress))
        else:
            tick = self._wall_tick(side)
            qty = self._scaled_quantity(1.00 + (0.35 * side_stress))

        self._book.add_limit(side, tick, qty)
        if side == "bid":
            self._bid_stress *= 0.92
        else:
            self._ask_stress *= 0.92
        self._refill_pressure *= 0.95
        self._invalidate_summary_state()

    def _apply_market_event(self) -> None:
        state = self._summary_state()
        aggressor: AggressorSide = "buy" if self._rng.random() < self._bid_probability(state) else "sell"
        depth_before = max(1, state.ask_depth if aggressor == "buy" else state.bid_depth)
        result = self._book.execute_market(aggressor, self._sample_quantity())
        if result.filled_qty <= 0:
            return

        if result.last_fill_tick is not None:
            self._last_trade_tick = result.last_fill_tick
        filled_qty = float(result.filled_qty)
        impact = clamp(filled_qty / float(depth_before), 0.0, 1.5)

        if aggressor == "buy":
            self._buy_aggr_volume += filled_qty
            self._flow_ema = clamp((0.82 * self._flow_ema) + (0.18 * impact), -1.5, 1.5)
            self._ask_stress = clamp((0.72 * self._ask_stress) + impact, 0.0, 3.0)
            self._bid_stress *= 0.90
        else:
            self._sell_aggr_volume += filled_qty
            self._flow_ema = clamp((0.82 * self._flow_ema) - (0.18 * impact), -1.5, 1.5)
            self._bid_stress = clamp((0.72 * self._bid_stress) + impact, 0.0, 3.0)
            self._ask_stress *= 0.90

        self._refill_pressure = clamp(max(self._refill_pressure * 0.85, impact), 0.0, 2.5)
        self._invalidate_summary_state()

    def _apply_cancel_event(self) -> None:
        state = self._summary_state()
        cancel_side: BookSide = "ask" if self._rng.random() < self._bid_probability(state) else "bid"
        side_stress = self._ask_stress if cancel_side == "ask" else self._bid_stress
        best_cancel_probability = clamp(0.20 + (0.45 * side_stress) + (0.20 * abs(self._pressure_signal(state))), 0.0, 0.90)

        if self._rng.random() < best_cancel_probability:
            tick = self._best_tick(cancel_side)
        else:
            tick = self._sample_existing_tick(cancel_side)
        if tick is None:
            return

        qty = min(self._scaled_quantity(0.90 + (0.40 * side_stress)), self._book.level_qty(cancel_side, tick))
        if qty <= 0:
            return
        self._book.cancel_level(cancel_side, tick, qty)
        self._invalidate_summary_state()

    def _sample_existing_tick(self, side: BookSide) -> int | None:
        levels = self._book.levels(side)
        if not levels:
            return None
        index = self._sample_level_index(len(levels))
        return levels[index][0]

    def _sample_level_index(self, size: int) -> int:
        weights = self._weights(size)
        return int(self._rng.choice(size, p=weights))

    def _weights(self, size: int):
        if size < 1:
            raise ValueError("weight size must be positive")
        cached = self._weight_cache.get(size)
        if cached is not None:
            return cached
        weights = np.power(self.config.level_decay, np.arange(size, dtype=float))
        normalized = weights / weights.sum()
        self._weight_cache[size] = normalized
        return normalized

    def _best_tick(self, side: BookSide) -> int | None:
        return self._book.best_bid_tick if side == "bid" else self._book.best_ask_tick

    def _inside_tick(self, side: BookSide) -> int:
        state = self._summary_state()
        if side == "bid":
            if state.spread_ticks > 1:
                return state.best_bid_tick + 1
            return state.best_bid_tick
        if state.spread_ticks > 1:
            return state.best_ask_tick - 1
        return state.best_ask_tick

    def _wall_tick(self, side: BookSide) -> int:
        state = self._summary_state()
        offset = 1 + self._sample_level_index(max(1, self._book_capacity - 1))
        if side == "bid":
            return max(0, min(state.best_bid_tick - offset, state.best_ask_tick - 1))
        return max(state.best_ask_tick + offset, state.best_bid_tick + 1)

    def _repair_book(self) -> None:
        self._ensure_two_sided()
        self._book.clear_crossed_quotes()
        self._ensure_two_sided()
        self._compress_spread()
        self._ensure_visible_levels()
        self._book.trim("bid", self._book_capacity)
        self._book.trim("ask", self._book_capacity)
        self._ensure_two_sided()
        self._invalidate_summary_state()

    def _ensure_two_sided(self) -> None:
        reference_tick = int(round(self._reference_tick()))
        best_bid = self._book.best_bid_tick
        best_ask = self._book.best_ask_tick

        if best_bid is None:
            target_bid = reference_tick - 1
            if best_ask is not None:
                target_bid = min(target_bid, best_ask - 1)
            self._book.add_limit("bid", target_bid, self._scaled_quantity(1.0 + self._refill_pressure))

        best_bid = self._book.best_bid_tick
        if best_ask is None:
            target_ask = reference_tick + 1
            if best_bid is not None:
                target_ask = max(target_ask, best_bid + 1)
            self._book.add_limit("ask", target_ask, self._scaled_quantity(1.0 + self._refill_pressure))

    def _compress_spread(self) -> None:
        state = self._summary_state()
        excess_spread = state.spread_ticks - self.config.max_spread_ticks
        if excess_spread <= 0:
            return
        for _ in range(excess_spread):
            state = self._summary_state()
            favored_bid = self._pressure_signal(state) >= 0.0
            if favored_bid:
                tick = self._inside_tick("bid")
                self._book.add_limit("bid", tick, self._scaled_quantity(1.0 + self._refill_pressure))
            else:
                tick = self._inside_tick("ask")
                self._book.add_limit("ask", tick, self._scaled_quantity(1.0 + self._refill_pressure))
            self._invalidate_summary_state()

    def _ensure_visible_levels(self) -> None:
        for side in ("bid", "ask"):
            missing = self.levels - self._book.level_count(side)
            for _ in range(max(0, missing)):
                next_tick = self._next_repair_tick(side)
                self._book.add_limit(side, next_tick, self._scaled_quantity(1.0 + self._refill_pressure))
        self._invalidate_summary_state()

    def _next_repair_tick(self, side: BookSide) -> int:
        reference_tick = int(round(self._reference_tick()))
        if side == "bid":
            deepest_bid = self._book.deepest_tick("bid")
            if deepest_bid is not None:
                return max(0, deepest_bid - 1)
            best_ask = self._book.best_ask_tick
            if best_ask is not None:
                return max(0, min(reference_tick - 1, best_ask - 1))
            return max(0, reference_tick - 1)

        deepest_ask = self._book.deepest_tick("ask")
        if deepest_ask is not None:
            return deepest_ask + 1
        best_bid = self._book.best_bid_tick
        if best_bid is not None:
            return max(reference_tick + 1, best_bid + 1)
        return reference_tick + 1

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
