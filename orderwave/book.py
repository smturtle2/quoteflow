from __future__ import annotations

"""Aggregate order-book primitives."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

BookSide = Literal["bid", "ask"]
AggressorSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class ExecutionResult:
    aggressor_side: AggressorSide
    requested_qty: int
    filled_qty: int
    fills: tuple[tuple[int, int], ...]

    @property
    def last_fill_tick(self) -> int | None:
        if not self.fills:
            return None
        return self.fills[-1][0]


class OrderBook:
    """Sparse aggregate order book keyed by integer ticks."""

    def __init__(self, tick_size: float) -> None:
        self.tick_size = float(tick_size)
        self._bids: dict[int, int] = {}
        self._asks: dict[int, int] = {}
        self._best_bid_tick: int | None = None
        self._best_ask_tick: int | None = None
        self._deepest_bid_tick: int | None = None
        self._deepest_ask_tick: int | None = None
        self._bid_levels_cache: tuple[tuple[int, int], ...] | None = None
        self._ask_levels_cache: tuple[tuple[int, int], ...] | None = None

    @property
    def best_bid_tick(self) -> int | None:
        return self._best_bid_tick

    @property
    def best_ask_tick(self) -> int | None:
        return self._best_ask_tick

    @property
    def spread_ticks(self) -> int | None:
        best_bid = self.best_bid_tick
        best_ask = self.best_ask_tick
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def has_side(self, side: BookSide) -> bool:
        return bool(self._book(side))

    def level_qty(self, side: BookSide, tick: int) -> int:
        return int(self._book(side).get(int(tick), 0))

    def set_level(self, side: BookSide, tick: int, qty: int) -> None:
        normalized_tick = max(0, int(tick))
        normalized_qty = int(qty)
        book = self._book(side)
        if normalized_qty <= 0:
            if normalized_tick in book:
                book.pop(normalized_tick, None)
                self._invalidate(side)
                self._refresh_extrema(side)
            return
        book[normalized_tick] = normalized_qty
        self._touch_tick(side, normalized_tick)

    def add_limit(self, side: BookSide, tick: int, qty: int) -> None:
        normalized_tick = max(0, int(tick))
        normalized_qty = int(qty)
        if normalized_qty <= 0:
            return
        book = self._book(side)
        book[normalized_tick] = int(book.get(normalized_tick, 0) + normalized_qty)
        self._touch_tick(side, normalized_tick)

    def cancel_level(self, side: BookSide, tick: int, qty: int) -> int:
        normalized_tick = max(0, int(tick))
        normalized_qty = int(qty)
        if normalized_qty <= 0:
            return 0
        book = self._book(side)
        resting = int(book.get(normalized_tick, 0))
        canceled = min(resting, normalized_qty)
        if canceled <= 0:
            return 0
        remaining = resting - canceled
        if remaining > 0:
            book[normalized_tick] = remaining
        else:
            book.pop(normalized_tick, None)
        self._invalidate(side)
        if remaining <= 0 and self._is_extrema_tick(side, normalized_tick):
            self._refresh_extrema(side)
        return canceled

    def execute_market(self, aggressor_side: AggressorSide, qty: int) -> ExecutionResult:
        requested_qty = int(qty)
        if requested_qty <= 0:
            return ExecutionResult(aggressor_side=aggressor_side, requested_qty=0, filled_qty=0, fills=())

        side: BookSide = "ask" if aggressor_side == "buy" else "bid"
        remaining = requested_qty
        fills: list[tuple[int, int]] = []
        while remaining > 0:
            best_tick = self.best_ask_tick if side == "ask" else self.best_bid_tick
            if best_tick is None:
                break
            resting = int(self._book(side)[best_tick])
            filled = min(resting, remaining)
            remaining -= filled
            if filled > 0:
                fills.append((best_tick, filled))
            updated = resting - filled
            if updated > 0:
                self._book(side)[best_tick] = updated
                self._invalidate(side)
            else:
                self._book(side).pop(best_tick, None)
                self._invalidate(side)
                self._refresh_extrema(side)

        return ExecutionResult(
            aggressor_side=aggressor_side,
            requested_qty=requested_qty,
            filled_qty=requested_qty - remaining,
            fills=tuple(fills),
        )

    def levels(self, side: BookSide, depth: int | None = None) -> tuple[tuple[int, int], ...]:
        cached = self._levels_cache(side)
        if depth is None:
            return cached
        return cached[: max(0, depth)]

    def total_depth(self, side: BookSide, depth: int | None = None) -> int:
        return sum(qty for _, qty in self.levels(side, depth))

    def level_count(self, side: BookSide) -> int:
        return len(self._book(side))

    def deepest_tick(self, side: BookSide) -> int | None:
        if side == "bid":
            return self._deepest_bid_tick
        return self._deepest_ask_tick

    def mid_tick(self) -> float | None:
        best_bid = self.best_bid_tick
        best_ask = self.best_ask_tick
        if best_bid is None or best_ask is None:
            return None
        return (float(best_bid) + float(best_ask)) / 2.0

    def trim(self, side: BookSide, max_levels: int) -> None:
        if max_levels < 1:
            self._book(side).clear()
            self._invalidate(side)
            self._refresh_extrema(side)
            return
        current_levels = self.levels(side)
        if len(current_levels) <= max_levels:
            return
        kept = dict(current_levels[:max_levels])
        book = self._book(side)
        book.clear()
        book.update(kept)
        self._invalidate(side)
        self._refresh_extrema(side)

    def clear_crossed_quotes(self) -> None:
        best_bid = self.best_bid_tick
        best_ask = self.best_ask_tick
        if best_bid is None or best_ask is None or best_bid < best_ask:
            return
        crossed_bids = [tick for tick in self._bids if tick >= best_ask]
        crossed_asks = [tick for tick in self._asks if tick <= best_bid]
        if crossed_bids:
            for tick in crossed_bids:
                self._bids.pop(tick, None)
            self._invalidate("bid")
            self._refresh_extrema("bid")
        if crossed_asks:
            for tick in crossed_asks:
                self._asks.pop(tick, None)
            self._invalidate("ask")
            self._refresh_extrema("ask")

    def signed_window(self, center_tick: int, window_ticks: int) -> np.ndarray:
        """Return signed depth around ``center_tick`` using relative price offsets."""

        width = (2 * window_ticks) + 1
        signed: np.ndarray = np.full(width, np.nan, dtype=np.float32)
        start_tick = int(center_tick) - int(window_ticks)
        end_tick = int(center_tick) + int(window_ticks)
        for tick, qty in self.levels("bid"):
            if tick < start_tick:
                break
            if tick > end_tick:
                continue
            signed[tick - start_tick] = float(qty)
        for tick, qty in self.levels("ask"):
            if tick > end_tick:
                break
            if tick < start_tick:
                continue
            signed[tick - start_tick] = -float(qty)
        return signed

    def _book(self, side: BookSide) -> dict[int, int]:
        return self._bids if side == "bid" else self._asks

    def _invalidate(self, side: BookSide) -> None:
        if side == "bid":
            self._bid_levels_cache = None
            return
        self._ask_levels_cache = None

    def _levels_cache(self, side: BookSide) -> tuple[tuple[int, int], ...]:
        if side == "bid":
            if self._bid_levels_cache is None:
                self._bid_levels_cache = tuple((tick, int(self._bids[tick])) for tick in sorted(self._bids, reverse=True))
            return self._bid_levels_cache
        if self._ask_levels_cache is None:
            self._ask_levels_cache = tuple((tick, int(self._asks[tick])) for tick in sorted(self._asks))
        return self._ask_levels_cache

    def _touch_tick(self, side: BookSide, tick: int) -> None:
        self._invalidate(side)
        if side == "bid":
            if self._best_bid_tick is None or tick > self._best_bid_tick:
                self._best_bid_tick = tick
            if self._deepest_bid_tick is None or tick < self._deepest_bid_tick:
                self._deepest_bid_tick = tick
            return
        if self._best_ask_tick is None or tick < self._best_ask_tick:
            self._best_ask_tick = tick
        if self._deepest_ask_tick is None or tick > self._deepest_ask_tick:
            self._deepest_ask_tick = tick

    def _refresh_extrema(self, side: BookSide) -> None:
        book = self._book(side)
        if side == "bid":
            if book:
                self._best_bid_tick = max(book)
                self._deepest_bid_tick = min(book)
            else:
                self._best_bid_tick = None
                self._deepest_bid_tick = None
            return
        if book:
            self._best_ask_tick = min(book)
            self._deepest_ask_tick = max(book)
        else:
            self._best_ask_tick = None
            self._deepest_ask_tick = None

    def _is_extrema_tick(self, side: BookSide, tick: int) -> bool:
        if side == "bid":
            return tick == self._best_bid_tick or tick == self._deepest_bid_tick
        return tick == self._best_ask_tick or tick == self._deepest_ask_tick


__all__ = ["AggressorSide", "BookSide", "ExecutionResult", "OrderBook"]
