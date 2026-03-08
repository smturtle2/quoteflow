from __future__ import annotations

import math

from orderwave.book import OrderBook


def ensure_two_sided_book(
    *,
    book: OrderBook,
    hidden_fair_tick: float,
    fallback_qty: int,
) -> None:
    if book.best_bid_tick is None:
        reference_tick = math.floor(hidden_fair_tick) - 1
        if book.best_ask_tick is not None:
            reference_tick = min(reference_tick, book.best_ask_tick - 1)
        book.add_limit("bid", max(0, int(reference_tick)), fallback_qty)

    if book.best_ask_tick is None:
        reference_tick = math.ceil(hidden_fair_tick) + 1
        if book.best_bid_tick is not None:
            reference_tick = max(reference_tick, book.best_bid_tick + 1)
        book.add_limit("ask", int(reference_tick), fallback_qty)


def ensure_visible_depth(
    *,
    book: OrderBook,
    minimum_visible_levels: int,
    fallback_qty: int,
) -> None:
    for side in ("bid", "ask"):
        current_levels = len(book.top_levels(side, minimum_visible_levels))
        for level in range(current_levels, minimum_visible_levels):
            book.apply_limit_relative(side, level, fallback_qty)


def apply_liquidity_backstop(
    *,
    book: OrderBook,
    mode: str,
    minimum_visible_levels: int,
    fallback_qty: int,
    hidden_fair_tick: float,
) -> None:
    if mode == "off":
        return
    if mode == "on_empty":
        if book.best_bid_tick is None or book.best_ask_tick is None:
            ensure_two_sided_book(
                book=book,
                hidden_fair_tick=hidden_fair_tick,
                fallback_qty=fallback_qty,
            )
        return
    ensure_two_sided_book(
        book=book,
        hidden_fair_tick=hidden_fair_tick,
        fallback_qty=fallback_qty,
    )
    ensure_visible_depth(
        book=book,
        minimum_visible_levels=minimum_visible_levels,
        fallback_qty=fallback_qty,
    )
