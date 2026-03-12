from __future__ import annotations

"""Aggregate order-book primitives."""

from dataclasses import dataclass
from typing import Literal

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

    @property
    def best_bid_tick(self) -> int | None:
        if not self._bids:
            return None
        return max(self._bids)

    @property
    def best_ask_tick(self) -> int | None:
        if not self._asks:
            return None
        return min(self._asks)

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
        book = self._book(side)
        normalized_tick = max(0, int(tick))
        normalized_qty = int(qty)
        if normalized_qty <= 0:
            book.pop(normalized_tick, None)
            return
        book[normalized_tick] = normalized_qty

    def add_limit(self, side: BookSide, tick: int, qty: int) -> None:
        normalized_tick = max(0, int(tick))
        normalized_qty = int(qty)
        if normalized_qty <= 0:
            return
        book = self._book(side)
        book[normalized_tick] = int(book.get(normalized_tick, 0) + normalized_qty)

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
        return canceled

    def execute_market(self, aggressor_side: AggressorSide, qty: int) -> ExecutionResult:
        requested_qty = int(qty)
        if requested_qty <= 0:
            return ExecutionResult(aggressor_side=aggressor_side, requested_qty=0, filled_qty=0, fills=())

        if aggressor_side == "buy":
            book = self._asks
            ticks = lambda: sorted(book)
        else:
            book = self._bids
            ticks = lambda: sorted(book, reverse=True)

        remaining = requested_qty
        fills: list[tuple[int, int]] = []
        for tick in ticks():
            if remaining <= 0:
                break
            resting = int(book[tick])
            filled = min(resting, remaining)
            remaining -= filled
            if filled > 0:
                fills.append((tick, filled))
            updated = resting - filled
            if updated > 0:
                book[tick] = updated
            else:
                book.pop(tick, None)

        return ExecutionResult(
            aggressor_side=aggressor_side,
            requested_qty=requested_qty,
            filled_qty=requested_qty - remaining,
            fills=tuple(fills),
        )

    def levels(self, side: BookSide, depth: int | None = None) -> tuple[tuple[int, int], ...]:
        book = self._book(side)
        reverse = side == "bid"
        ticks = sorted(book, reverse=reverse)
        if depth is not None:
            ticks = ticks[: max(0, depth)]
        return tuple((tick, int(book[tick])) for tick in ticks)

    def total_depth(self, side: BookSide, depth: int | None = None) -> int:
        return sum(qty for _, qty in self.levels(side, depth))

    def level_count(self, side: BookSide) -> int:
        return len(self._book(side))

    def deepest_tick(self, side: BookSide) -> int | None:
        levels = self.levels(side)
        if not levels:
            return None
        return levels[-1][0]

    def mid_tick(self) -> float | None:
        best_bid = self.best_bid_tick
        best_ask = self.best_ask_tick
        if best_bid is None or best_ask is None:
            return None
        return (float(best_bid) + float(best_ask)) / 2.0

    def trim(self, side: BookSide, max_levels: int) -> None:
        if max_levels < 1:
            self._book(side).clear()
            return
        kept = dict(self.levels(side, max_levels))
        book = self._book(side)
        book.clear()
        book.update(kept)

    def clear_crossed_quotes(self) -> None:
        best_bid = self.best_bid_tick
        best_ask = self.best_ask_tick
        if best_bid is None or best_ask is None or best_bid < best_ask:
            return
        crossed_bids = [tick for tick in self._bids if tick >= best_ask]
        crossed_asks = [tick for tick in self._asks if tick <= best_bid]
        for tick in crossed_bids:
            self._bids.pop(tick, None)
        for tick in crossed_asks:
            self._asks.pop(tick, None)

    def _book(self, side: BookSide) -> dict[int, int]:
        return self._bids if side == "bid" else self._asks


__all__ = ["AggressorSide", "BookSide", "ExecutionResult", "OrderBook"]
