from __future__ import annotations

from orderwave.book import OrderBook


def test_market_buy_sweeps_multiple_ask_levels() -> None:
    book = OrderBook(tick_size=0.01)
    book.set_level("bid", 10000, 6)
    book.set_level("ask", 10001, 3)
    book.set_level("ask", 10002, 4)

    result = book.execute_market("buy", 5)

    assert result.filled_qty == 5
    assert result.last_fill_tick == 10002
    assert book.best_ask_tick == 10002
    assert book.ask_book[10002] == 2


def test_cancel_never_exceeds_resting_qty() -> None:
    book = OrderBook(tick_size=0.01)
    book.set_level("bid", 10000, 5)
    book.set_level("ask", 10001, 6)

    canceled = book.cancel_level("ask", 10001, 20)

    assert canceled == 6
    assert book.best_ask_tick is None


def test_inside_spread_limit_improves_quote_without_crossing() -> None:
    book = OrderBook(tick_size=0.01)
    book.set_level("bid", 10000, 5)
    book.set_level("ask", 10003, 5)

    applied_tick = book.apply_limit_relative("bid", -1, 4)

    assert applied_tick == 10001
    assert book.best_bid_tick == 10001
    assert book.best_bid_tick < book.best_ask_tick
