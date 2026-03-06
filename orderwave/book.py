from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BookSide = Literal["bid", "ask"]
AggressorSide = Literal["buy", "sell"]


@dataclass
class ExecutionResult:
    aggressor_side: AggressorSide
    requested_qty: int
    filled_qty: int = 0
    last_fill_tick: int | None = None
    fills: list[tuple[int, int]] = field(default_factory=list)


class OrderBook:
    def __init__(self, tick_size: float) -> None:
        self.tick_size = float(tick_size)
        self.bid_book: dict[int, int] = {}
        self.ask_book: dict[int, int] = {}
        self.bid_staleness: dict[int, int] = {}
        self.ask_staleness: dict[int, int] = {}
        self.best_bid_tick: int | None = None
        self.best_ask_tick: int | None = None

    @property
    def spread_ticks(self) -> int:
        if self.best_bid_tick is None or self.best_ask_tick is None:
            return 0
        return self.best_ask_tick - self.best_bid_tick

    def set_level(self, side: BookSide, tick: int, qty: int) -> None:
        book, staleness = self._book_and_age(side)
        if qty <= 0:
            book.pop(tick, None)
            staleness.pop(tick, None)
            self._refresh_best(side)
            return
        book[tick] = int(qty)
        staleness[tick] = 0
        self._refresh_best(side)

    def add_limit(self, side: BookSide, tick: int, qty: int) -> None:
        if qty <= 0:
            return
        book, staleness = self._book_and_age(side)
        book[tick] = int(book.get(tick, 0) + qty)
        staleness[tick] = 0
        self._refresh_best(side)

    def apply_limit_relative(self, side: BookSide, level: int, qty: int) -> int | None:
        tick = self.resolve_limit_tick(side, level)
        if tick is None:
            return None
        self.add_limit(side, tick, qty)
        return tick

    def resolve_limit_tick(self, side: BookSide, level: int) -> int | None:
        if self.best_bid_tick is None or self.best_ask_tick is None:
            return None

        if side == "bid":
            if level == -1:
                if self.spread_ticks <= 1:
                    return None
                target_tick = self.best_bid_tick + 1
            else:
                target_tick = self.best_bid_tick - int(level)
            if target_tick < 0 or target_tick >= self.best_ask_tick:
                return None
            return target_tick

        if level == -1:
            if self.spread_ticks <= 1:
                return None
            target_tick = self.best_ask_tick - 1
        else:
            target_tick = self.best_ask_tick + int(level)
        if target_tick <= self.best_bid_tick:
            return None
        return target_tick

    def cancel_level(self, side: BookSide, tick: int, qty: int) -> int:
        if qty <= 0:
            return 0
        book, staleness = self._book_and_age(side)
        resting = book.get(tick, 0)
        canceled = min(resting, int(qty))
        if canceled <= 0:
            return 0
        remaining = resting - canceled
        if remaining > 0:
            book[tick] = remaining
            staleness[tick] = 0
        else:
            book.pop(tick, None)
            staleness.pop(tick, None)
        self._refresh_best(side)
        return canceled

    def execute_market(self, aggressor_side: AggressorSide, qty: int) -> ExecutionResult:
        result = ExecutionResult(aggressor_side=aggressor_side, requested_qty=int(qty))
        if qty <= 0:
            return result

        if aggressor_side == "buy":
            book = self.ask_book
            staleness = self.ask_staleness
            side = "ask"
            best_key = lambda: self.best_ask_tick
        else:
            book = self.bid_book
            staleness = self.bid_staleness
            side = "bid"
            best_key = lambda: self.best_bid_tick

        remaining = int(qty)
        while remaining > 0:
            best_tick = best_key()
            if best_tick is None:
                break

            resting = book[best_tick]
            taken = min(resting, remaining)
            remaining -= taken
            result.filled_qty += taken
            result.last_fill_tick = best_tick
            result.fills.append((best_tick, taken))

            new_qty = resting - taken
            if new_qty > 0:
                book[best_tick] = new_qty
                staleness[best_tick] = 0
            else:
                book.pop(best_tick, None)
                staleness.pop(best_tick, None)
                self._refresh_best(side)

        return result

    def increment_staleness(self) -> None:
        for staleness in (self.bid_staleness, self.ask_staleness):
            for tick in list(staleness):
                staleness[tick] += 1

    def top_levels(self, side: BookSide, depth: int) -> list[tuple[int, int]]:
        return self.all_levels(side)[: max(0, depth)]

    def all_levels(self, side: BookSide) -> list[tuple[int, int]]:
        book, _ = self._book_and_age(side)
        ticks = sorted(book, reverse=(side == "bid"))
        return [(tick, int(book[tick])) for tick in ticks]

    def best_qty(self, side: BookSide) -> int:
        if side == "bid":
            if self.best_bid_tick is None:
                return 0
            return int(self.bid_book[self.best_bid_tick])
        if self.best_ask_tick is None:
            return 0
        return int(self.ask_book[self.best_ask_tick])

    def total_depth(self, side: BookSide, depth: int) -> int:
        return sum(qty for _, qty in self.top_levels(side, depth))

    def level_age(self, side: BookSide, tick: int) -> int:
        _, staleness = self._book_and_age(side)
        return int(staleness.get(tick, 0))

    def _book_and_age(self, side: BookSide) -> tuple[dict[int, int], dict[int, int]]:
        if side == "bid":
            return self.bid_book, self.bid_staleness
        return self.ask_book, self.ask_staleness

    def _refresh_best(self, side: BookSide) -> None:
        if side == "bid":
            self.best_bid_tick = max(self.bid_book) if self.bid_book else None
            return
        self.best_ask_tick = min(self.ask_book) if self.ask_book else None
