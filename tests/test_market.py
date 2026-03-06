from __future__ import annotations

import pandas as pd

from orderwave import Market
from orderwave.book import OrderBook
from orderwave.metrics import compute_features


def test_market_initializes_snapshot_and_history() -> None:
    market = Market(seed=42)

    snapshot = market.get()
    history = market.get_history()

    assert snapshot["step"] == 0
    assert snapshot["last_price"] == 100.0
    assert snapshot["best_bid"] < snapshot["best_ask"]
    assert len(snapshot["bids"]) <= market.levels
    assert len(snapshot["asks"]) <= market.levels
    assert list(history["step"]) == [0]


def test_gen_matches_repeated_step_for_same_seed() -> None:
    batch_market = Market(seed=42)
    step_market = Market(seed=42)

    batch_snapshot = batch_market.gen(steps=25)
    for _ in range(25):
        repeated_snapshot = step_market.step()

    assert batch_snapshot == repeated_snapshot
    pd.testing.assert_frame_equal(batch_market.get_history(), step_market.get_history())
    pd.testing.assert_frame_equal(batch_market.get_event_history(), step_market.get_event_history())


def test_same_seed_reproduces_and_other_seed_differs() -> None:
    market_a = Market(seed=11)
    market_b = Market(seed=11)
    market_c = Market(seed=12)

    market_a.gen(steps=100)
    market_b.gen(steps=100)
    market_c.gen(steps=100)

    pd.testing.assert_frame_equal(market_a.get_history(), market_b.get_history())
    assert not market_a.get_event_history().equals(market_c.get_event_history())


def test_history_contains_summary_columns_only() -> None:
    market = Market(seed=7)
    market.gen(steps=5)

    history = market.get_history()

    expected = {
        "step",
        "last_price",
        "mid_price",
        "microprice",
        "best_bid",
        "best_ask",
        "spread",
        "buy_aggr_volume",
        "sell_aggr_volume",
        "trade_strength",
        "depth_imbalance",
        "regime",
        "top_n_bid_qty",
        "top_n_ask_qty",
        "realized_vol",
        "signed_flow",
    }
    assert set(history.columns) == expected
    assert "bids" not in history.columns
    assert "asks" not in history.columns


def test_book_invariants_hold_over_random_run() -> None:
    market = Market(seed=9, config={"preset": "volatile"})
    market.gen(steps=100)

    snapshot = market.get()
    bid_prices = [level["price"] for level in snapshot["bids"]]
    ask_prices = [level["price"] for level in snapshot["asks"]]

    assert bid_prices == sorted(bid_prices, reverse=True)
    assert ask_prices == sorted(ask_prices)
    assert snapshot["best_bid"] < snapshot["best_ask"]
    assert all(level["qty"] > 0 for level in snapshot["bids"])
    assert all(level["qty"] > 0 for level in snapshot["asks"])


def test_quote_only_book_change_can_move_mid_without_touching_last_price() -> None:
    market = Market(seed=1)
    market._book = OrderBook(tick_size=market.tick_size)
    market._book.set_level("bid", 10000, 5)
    market._book.set_level("ask", 10003, 5)
    market._buy_flow.clear()
    market._sell_flow.clear()
    market._mid_returns.clear()

    before = market._build_snapshot(
        compute_features(
            market._book,
            tick_size=market.tick_size,
            depth_levels=market.levels,
            buy_flow=[],
            sell_flow=[],
            mid_returns=[],
        )
    )

    market._book.cancel_level("ask", 10003, 5)
    market._book.set_level("ask", 10004, 5)
    after = market._build_snapshot(
        compute_features(
            market._book,
            tick_size=market.tick_size,
            depth_levels=market.levels,
            buy_flow=[],
            sell_flow=[],
            mid_returns=[],
        )
    )

    assert before["mid_price"] != after["mid_price"]
    assert before["last_price"] == after["last_price"] == 100.0


def test_trade_updates_last_price_and_trade_metadata() -> None:
    market = Market(seed=2)
    market._book = OrderBook(tick_size=market.tick_size)
    market._book.set_level("bid", 10000, 6)
    market._book.set_level("ask", 10001, 2)
    market._book.set_level("ask", 10002, 3)

    result = market._book.execute_market("buy", 4)
    market._last_trade_price = market.tick_size * result.last_fill_tick
    market._last_trade_side = "buy"
    market._last_trade_qty = float(result.filled_qty)

    assert result.last_fill_tick == 10002
    assert market._last_trade_price == 100.02
    assert market._last_trade_side == "buy"
    assert market._last_trade_qty == 4.0
