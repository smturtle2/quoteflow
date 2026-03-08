from __future__ import annotations

from orderwave._model.backstop import apply_liquidity_backstop
from orderwave.book import OrderBook


def test_liquidity_backstop_always_restores_visible_depth() -> None:
    book = OrderBook(tick_size=0.01)
    book.set_level("bid", 10000, 2)
    book.set_level("ask", 10001, 2)

    apply_liquidity_backstop(
        book=book,
        mode="always",
        minimum_visible_levels=3,
        fallback_qty=5,
        hidden_fair_tick=10000.5,
    )

    assert len(book.top_levels("bid", 3)) == 3
    assert len(book.top_levels("ask", 3)) == 3


def test_liquidity_backstop_on_empty_only_restores_missing_side() -> None:
    book = OrderBook(tick_size=0.01)
    book.set_level("bid", 10000, 2)
    book.set_level("ask", 10001, 2)

    apply_liquidity_backstop(
        book=book,
        mode="on_empty",
        minimum_visible_levels=3,
        fallback_qty=5,
        hidden_fair_tick=10000.5,
    )
    assert len(book.top_levels("bid", 3)) == 1
    assert len(book.top_levels("ask", 3)) == 1

    book.cancel_level("ask", 10001, 2)
    apply_liquidity_backstop(
        book=book,
        mode="on_empty",
        minimum_visible_levels=3,
        fallback_qty=5,
        hidden_fair_tick=10000.5,
    )

    assert book.best_ask_tick is not None
    assert len(book.top_levels("bid", 3)) == 1


def test_liquidity_backstop_off_leaves_empty_side_empty() -> None:
    book = OrderBook(tick_size=0.01)
    book.set_level("bid", 10000, 2)
    book.set_level("ask", 10001, 2)
    book.cancel_level("ask", 10001, 2)

    apply_liquidity_backstop(
        book=book,
        mode="off",
        minimum_visible_levels=3,
        fallback_qty=5,
        hidden_fair_tick=10000.5,
    )

    assert book.best_ask_tick is None
