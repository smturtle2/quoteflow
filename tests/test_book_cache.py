from __future__ import annotations

import numpy as np

from orderwave.book import OrderBook


def test_book_cache_and_signed_window_stay_consistent() -> None:
    book = OrderBook(tick_size=0.01)
    book.add_limit("bid", 99, 5)
    book.add_limit("bid", 98, 4)
    book.add_limit("ask", 101, 6)
    book.add_limit("ask", 102, 3)

    assert book.best_bid_tick == 99
    assert book.best_ask_tick == 101
    assert book.deepest_tick("bid") == 98
    assert book.deepest_tick("ask") == 102
    assert book.levels("bid") == ((99, 5), (98, 4))
    assert book.levels("ask") == ((101, 6), (102, 3))

    book.add_limit("bid", 100, 2)
    book.cancel_level("ask", 101, 6)
    book.add_limit("ask", 103, 7)

    assert book.best_bid_tick == 100
    assert book.best_ask_tick == 102
    assert book.deepest_tick("bid") == 98
    assert book.deepest_tick("ask") == 103

    signed = book.signed_window(center_tick=101, window_ticks=3)
    expected = np.array([4.0, 5.0, 2.0, np.nan, -3.0, -7.0, np.nan], dtype=np.float32)
    np.testing.assert_allclose(signed, expected, equal_nan=True)
