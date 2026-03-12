from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from orderwave.utils import clamp

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
        self.bid_reprice_pressure: dict[int, float] = {}
        self.ask_reprice_pressure: dict[int, float] = {}
        self.bid_refill_propensity: dict[int, float] = {}
        self.ask_refill_propensity: dict[int, float] = {}
        self.bid_toxicity: dict[int, float] = {}
        self.ask_toxicity: dict[int, float] = {}
        self.best_bid_tick: int | None = None
        self.best_ask_tick: int | None = None
        self._bid_levels_cache: tuple[tuple[int, int], ...] | None = None
        self._ask_levels_cache: tuple[tuple[int, int], ...] | None = None

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
            self._remove_level_state(side, tick)
            self._invalidate_levels_cache(side)
            self._refresh_best(side)
            return
        book[tick] = int(qty)
        staleness[tick] = 0
        self._ensure_level_state(side, tick)
        self._invalidate_levels_cache(side)
        self._refresh_best(side)

    def add_limit(self, side: BookSide, tick: int, qty: int) -> None:
        if qty <= 0:
            return
        book, staleness = self._book_and_age(side)
        book[tick] = int(book.get(tick, 0) + qty)
        staleness[tick] = 0
        self._ensure_level_state(side, tick)
        self._nudge_after_refill(side, tick)
        self._invalidate_levels_cache(side)
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
        cancel_ratio = canceled / max(resting, 1)
        self._note_quote_revision(side, tick, cancel_ratio)
        remaining = resting - canceled
        if remaining > 0:
            book[tick] = remaining
            staleness[tick] = 0
        else:
            book.pop(tick, None)
            staleness.pop(tick, None)
            self._remove_level_state(side, tick)
        self._invalidate_levels_cache(side)
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
            self._note_execution(side, best_tick, taken / max(resting, 1))

            new_qty = resting - taken
            if new_qty > 0:
                book[best_tick] = new_qty
                staleness[best_tick] = 0
            else:
                book.pop(best_tick, None)
                staleness.pop(best_tick, None)
                self._remove_level_state(side, best_tick)
            self._invalidate_levels_cache(side)
            self._refresh_best(side)

        return result

    def increment_staleness(self) -> None:
        for staleness in (self.bid_staleness, self.ask_staleness):
            for tick in list(staleness):
                staleness[tick] += 1
        for values in (
            self.bid_reprice_pressure,
            self.ask_reprice_pressure,
            self.bid_refill_propensity,
            self.ask_refill_propensity,
            self.bid_toxicity,
            self.ask_toxicity,
        ):
            for tick in list(values):
                values[tick] = float(values[tick] * 0.92)
                if values[tick] < 1e-3:
                    values.pop(tick, None)

    def top_levels(self, side: BookSide, depth: int) -> tuple[tuple[int, int], ...]:
        return self.all_levels(side)[: max(0, depth)]

    def all_levels(self, side: BookSide) -> tuple[tuple[int, int], ...]:
        cached = self._bid_levels_cache if side == "bid" else self._ask_levels_cache
        if cached is None:
            book, _ = self._book_and_age(side)
            cached = tuple((tick, int(book[tick])) for tick in sorted(book, reverse=(side == "bid")))
            if side == "bid":
                self._bid_levels_cache = cached
            else:
                self._ask_levels_cache = cached
        return cached

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

    def top_depth_state(self, depth: int) -> tuple[int, int, int, int]:
        if depth <= 0:
            return self.best_qty("bid"), self.best_qty("ask"), 0, 0
        bid_levels = self.top_levels("bid", depth)
        ask_levels = self.top_levels("ask", depth)
        bid_total = sum(qty for _, qty in bid_levels)
        ask_total = sum(qty for _, qty in ask_levels)
        best_bid_qty = bid_levels[0][1] if bid_levels else 0
        best_ask_qty = ask_levels[0][1] if ask_levels else 0
        return int(best_bid_qty), int(best_ask_qty), int(bid_total), int(ask_total)

    def level_age(self, side: BookSide, tick: int) -> int:
        _, staleness = self._book_and_age(side)
        return int(staleness.get(tick, 0))

    def level_reprice_pressure(self, side: BookSide, tick: int) -> float:
        reprice, _, _ = self._state_maps(side)
        return float(reprice.get(tick, 0.0))

    def level_refill_propensity(self, side: BookSide, tick: int) -> float:
        _, refill, _ = self._state_maps(side)
        return float(refill.get(tick, 0.0))

    def level_toxicity(self, side: BookSide, tick: int) -> float:
        _, _, toxicity = self._state_maps(side)
        return float(toxicity.get(tick, 0.0))

    def _book_and_age(self, side: BookSide) -> tuple[dict[int, int], dict[int, int]]:
        if side == "bid":
            return self.bid_book, self.bid_staleness
        return self.ask_book, self.ask_staleness

    def _state_maps(self, side: BookSide) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        if side == "bid":
            return self.bid_reprice_pressure, self.bid_refill_propensity, self.bid_toxicity
        return self.ask_reprice_pressure, self.ask_refill_propensity, self.ask_toxicity

    def _ensure_level_state(self, side: BookSide, tick: int) -> None:
        reprice, refill, toxicity = self._state_maps(side)
        reprice.setdefault(int(tick), 0.0)
        refill.setdefault(int(tick), 0.0)
        toxicity.setdefault(int(tick), 0.0)

    def _remove_level_state(self, side: BookSide, tick: int) -> None:
        reprice, refill, toxicity = self._state_maps(side)
        reprice.pop(int(tick), None)
        refill.pop(int(tick), None)
        toxicity.pop(int(tick), None)

    def _nudge_after_refill(self, side: BookSide, tick: int) -> None:
        reprice, refill, toxicity = self._state_maps(side)
        reprice[int(tick)] = float(clamp(reprice.get(int(tick), 0.0) * 0.7, 0.0, 6.0))
        refill[int(tick)] = float(clamp(refill.get(int(tick), 0.0) * 0.55, 0.0, 6.0))
        toxicity[int(tick)] = float(clamp(toxicity.get(int(tick), 0.0) * 0.82, 0.0, 6.0))

    def _note_quote_revision(self, side: BookSide, tick: int, magnitude: float) -> None:
        self._ensure_level_state(side, tick)
        reprice, refill, toxicity = self._state_maps(side)
        reprice[int(tick)] = float(clamp(reprice.get(int(tick), 0.0) + (0.8 * magnitude), 0.0, 6.0))
        refill[int(tick)] = float(clamp(refill.get(int(tick), 0.0) + (0.35 * magnitude), 0.0, 6.0))
        toxicity[int(tick)] = float(clamp(toxicity.get(int(tick), 0.0) + (0.12 * magnitude), 0.0, 6.0))

    def _note_execution(self, side: BookSide, tick: int, magnitude: float) -> None:
        self._ensure_level_state(side, tick)
        reprice, refill, toxicity = self._state_maps(side)
        reprice[int(tick)] = float(clamp(reprice.get(int(tick), 0.0) + (0.55 * magnitude), 0.0, 6.0))
        refill[int(tick)] = float(clamp(refill.get(int(tick), 0.0) + (0.75 * magnitude), 0.0, 6.0))
        toxicity[int(tick)] = float(clamp(toxicity.get(int(tick), 0.0) + (1.15 * magnitude), 0.0, 6.0))

    def _refresh_best(self, side: BookSide) -> None:
        if side == "bid":
            self.best_bid_tick = max(self.bid_book) if self.bid_book else None
            return
        self.best_ask_tick = min(self.ask_book) if self.ask_book else None

    def _invalidate_levels_cache(self, side: BookSide) -> None:
        if side == "bid":
            self._bid_levels_cache = None
        else:
            self._ask_levels_cache = None
