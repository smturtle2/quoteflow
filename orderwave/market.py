from __future__ import annotations

"""Public market simulator API."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from orderwave.book import AggressorSide, BookSide, OrderBook
from orderwave.config import MarketConfig, coerce_config
from orderwave.history import HistoryBuffer
from orderwave.utils import bounded_int, clamp, compute_depth_imbalance, price_to_tick, tick_to_price


@dataclass(frozen=True)
class SimulationResult:
    """Collected outputs returned by ``Market.run``."""

    snapshot: dict[str, object]
    history: pd.DataFrame


class Market:
    """Compact aggregate order-book simulator driven by explicit distributions."""

    def __init__(
        self,
        init_price: float = 100.0,
        tick_size: float = 0.01,
        levels: int = 5,
        seed: int | None = None,
        config: MarketConfig | Mapping[str, object] | None = None,
    ) -> None:
        if tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        if levels <= 0:
            raise ValueError("levels must be positive")

        self.tick_size = float(tick_size)
        self.levels = int(levels)
        self.seed = seed
        self.config = coerce_config(config)
        self._rng = np.random.default_rng(seed)

        self._book = OrderBook(self.tick_size)
        self._history = HistoryBuffer()
        self._step = 0
        self._book_capacity = self.levels + self.config.max_spread_ticks
        self._weight_cache: dict[int, np.ndarray] = {}

        self._init_tick = price_to_tick(init_price, self.tick_size)
        self._last_trade_tick = self._init_tick
        self._fair_tick = float(self._init_tick)
        self._buy_aggr_volume = 0.0
        self._sell_aggr_volume = 0.0

        self._seed_book(self._init_tick)
        self._record_history()

    def get(self) -> dict[str, object]:
        """Return the current public market snapshot."""

        best_bid_tick = self._book.best_bid_tick
        best_ask_tick = self._book.best_ask_tick
        if best_bid_tick is None or best_ask_tick is None:
            raise RuntimeError("order book must remain two-sided")

        bid_depth = self._book.total_depth("bid", self.levels)
        ask_depth = self._book.total_depth("ask", self.levels)
        spread_ticks = best_ask_tick - best_bid_tick

        return {
            "step": self._step,
            "last_price": tick_to_price(self._last_trade_tick, self.tick_size),
            "mid_price": tick_to_price((best_bid_tick + best_ask_tick) / 2.0, self.tick_size),
            "best_bid": tick_to_price(best_bid_tick, self.tick_size),
            "best_ask": tick_to_price(best_ask_tick, self.tick_size),
            "spread": tick_to_price(spread_ticks, self.tick_size),
            "bids": self._public_levels("bid"),
            "asks": self._public_levels("ask"),
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "depth_imbalance": compute_depth_imbalance(bid_depth, ask_depth),
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

        self._advance_fair_price()
        event_types = self._sample_event_types()
        self._rng.shuffle(event_types)
        for event_type in event_types:
            if event_type == "limit":
                self._apply_limit_event()
            elif event_type == "market":
                self._apply_market_event()
            else:
                self._apply_cancel_event()

        self._repair_book()
        self._record_history()
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

    def _seed_book(self, center_tick: int) -> None:
        for offset in range(self._book_capacity):
            self._book.add_limit("bid", center_tick - 1 - offset, self._sample_quantity())
            self._book.add_limit("ask", center_tick + 1 + offset, self._sample_quantity())

    def _record_history(self) -> None:
        self._history.append(self.get())

    def _coerce_steps(self, steps: int) -> int:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return int(steps)

    def _advance_fair_price(self) -> None:
        reference_tick = self._reference_tick()
        reversion = self.config.mean_reversion * (reference_tick - self._fair_tick)
        shock = float(self._rng.normal(0.0, self.config.fair_price_vol))
        move = clamp(
            reversion + shock,
            -float(self.config.max_fair_move_ticks),
            float(self.config.max_fair_move_ticks),
        )
        self._fair_tick += move

    def _sample_event_types(self) -> list[str]:
        event_types = ["limit"] * int(self._rng.poisson(self.config.limit_rate))
        event_types.extend(["market"] * int(self._rng.poisson(self.config.market_rate)))
        event_types.extend(["cancel"] * int(self._rng.poisson(self.config.cancel_rate)))
        return event_types

    def _sample_quantity(self) -> int:
        raw_value = float(self._rng.lognormal(self.config.size_mean, self.config.size_dispersion))
        return bounded_int(raw_value, self.config.min_order_qty, self.config.max_order_qty)

    def _sample_probability_for_bid(self) -> float:
        mid_tick = self._reference_tick()
        fair_signal = clamp(
            (self._fair_tick - mid_tick) / float(self.config.max_fair_move_ticks),
            -1.0,
            1.0,
        )
        imbalance = self._current_imbalance()
        pressure = clamp((fair_signal - imbalance) / 2.0, -1.0, 1.0)
        return 0.5 * (1.0 + pressure)

    def _apply_limit_event(self) -> None:
        bid_probability = self._sample_probability_for_bid()
        fair_signal = clamp(
            (self._fair_tick - self._reference_tick()) / float(self.config.max_fair_move_ticks),
            -1.0,
            1.0,
        )
        side: BookSide = "bid" if self._rng.random() < bid_probability else "ask"
        offset = self._sample_level_index(self._book_capacity)
        tick = self._resolve_limit_tick(side, offset, fair_signal)
        self._book.add_limit(side, tick, self._sample_quantity())

    def _apply_market_event(self) -> None:
        bid_probability = self._sample_probability_for_bid()
        aggressor: AggressorSide = "buy" if self._rng.random() < bid_probability else "sell"
        result = self._book.execute_market(aggressor, self._sample_quantity())
        if result.filled_qty <= 0:
            return
        if result.last_fill_tick is not None:
            self._last_trade_tick = result.last_fill_tick
        if aggressor == "buy":
            self._buy_aggr_volume += float(result.filled_qty)
        else:
            self._sell_aggr_volume += float(result.filled_qty)

    def _apply_cancel_event(self) -> None:
        bid_probability = self._sample_probability_for_bid()
        side: BookSide = "ask" if self._rng.random() < bid_probability else "bid"
        tick = self._sample_existing_tick(side)
        if tick is None:
            fallback_side: BookSide = "bid" if side == "ask" else "ask"
            tick = self._sample_existing_tick(fallback_side)
            side = fallback_side
        if tick is None:
            return
        qty = min(self._sample_quantity(), self._book.level_qty(side, tick))
        self._book.cancel_level(side, tick, qty)

    def _sample_existing_tick(self, side: BookSide) -> int | None:
        levels = self._book.levels(side)
        if not levels:
            return None
        index = self._sample_level_index(len(levels))
        return levels[index][0]

    def _sample_level_index(self, size: int) -> int:
        weights = self._weights(size)
        return int(self._rng.choice(size, p=weights))

    def _weights(self, size: int) -> np.ndarray:
        if size < 1:
            raise ValueError("weight size must be positive")
        cached = self._weight_cache.get(size)
        if cached is not None:
            return cached
        weights = np.power(self.config.level_decay, np.arange(size, dtype=float))
        normalized = weights / weights.sum()
        self._weight_cache[size] = normalized
        return normalized

    def _resolve_limit_tick(self, side: BookSide, offset: int, fair_signal: float) -> int:
        best_bid = self._book.best_bid_tick
        best_ask = self._book.best_ask_tick
        spread_ticks = self._book.spread_ticks or 0

        if best_bid is not None and best_ask is not None:
            favored = (side == "bid" and fair_signal > 0.0) or (side == "ask" and fair_signal < 0.0)
            if spread_ticks > 1 and favored and self._rng.random() < abs(fair_signal):
                return best_bid + 1 if side == "bid" else best_ask - 1
            if side == "bid":
                return max(0, min(best_bid - offset, best_ask - 1))
            return max(best_ask + offset, best_bid + 1)

        reference_tick = int(round(self._reference_tick()))
        if side == "bid":
            if best_ask is not None:
                return max(0, min(reference_tick - 1 - offset, best_ask - 1))
            return max(0, reference_tick - 1 - offset)
        if best_bid is not None:
            return max(reference_tick + 1 + offset, best_bid + 1)
        return reference_tick + 1 + offset

    def _repair_book(self) -> None:
        self._ensure_two_sided()
        self._book.clear_crossed_quotes()
        self._ensure_two_sided()
        self._compress_spread()
        self._ensure_visible_levels()
        self._book.trim("bid", self._book_capacity)
        self._book.trim("ask", self._book_capacity)
        self._ensure_two_sided()

    def _ensure_two_sided(self) -> None:
        reference_tick = int(round(self._reference_tick()))
        best_bid = self._book.best_bid_tick
        best_ask = self._book.best_ask_tick

        if best_bid is None:
            target_bid = reference_tick - 1
            if best_ask is not None:
                target_bid = min(target_bid, best_ask - 1)
            self._book.add_limit("bid", target_bid, self._sample_quantity())

        best_bid = self._book.best_bid_tick
        if best_ask is None:
            target_ask = reference_tick + 1
            if best_bid is not None:
                target_ask = max(target_ask, best_bid + 1)
            self._book.add_limit("ask", target_ask, self._sample_quantity())

    def _compress_spread(self) -> None:
        while True:
            spread_ticks = self._book.spread_ticks
            if spread_ticks is None or spread_ticks <= self.config.max_spread_ticks:
                return
            reference_tick = self._reference_tick()
            if self._fair_tick >= reference_tick:
                best_bid = self._book.best_bid_tick
                best_ask = self._book.best_ask_tick
                if best_bid is None or best_ask is None:
                    return
                self._book.add_limit("bid", min(best_bid + 1, best_ask - 1), self._sample_quantity())
            else:
                best_bid = self._book.best_bid_tick
                best_ask = self._book.best_ask_tick
                if best_bid is None or best_ask is None:
                    return
                self._book.add_limit("ask", max(best_ask - 1, best_bid + 1), self._sample_quantity())

    def _ensure_visible_levels(self) -> None:
        for side in ("bid", "ask"):
            attempts = 0
            while self._book.level_count(side) < self.levels and attempts < self._book_capacity:
                next_tick = self._next_repair_tick(side)
                before = self._book.level_count(side)
                self._book.add_limit(side, next_tick, self._sample_quantity())
                after = self._book.level_count(side)
                if after == before:
                    break
                attempts += 1

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

    def _current_imbalance(self) -> float:
        bid_depth = self._book.total_depth("bid", self.levels)
        ask_depth = self._book.total_depth("ask", self.levels)
        return compute_depth_imbalance(bid_depth, ask_depth)


__all__ = ["Market", "SimulationResult"]
