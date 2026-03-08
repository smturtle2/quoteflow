from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from orderwave import Market


def _stable_frame_hash(frame: pd.DataFrame) -> str:
    payload = {
        "columns": list(frame.columns),
        "records": frame.to_dict(orient="records"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_market_initializes_snapshot_and_history() -> None:
    market = Market(seed=42)

    snapshot = market.get()
    history = market.get_history()

    assert snapshot["step"] == 0
    assert snapshot["day"] == 0
    assert snapshot["session_step"] == 0
    assert snapshot["session_phase"] == "open"
    assert snapshot["last_price"] == 100.0
    assert snapshot["best_bid"] < snapshot["best_ask"]
    assert len(snapshot["bids"]) <= market.levels
    assert len(snapshot["asks"]) <= market.levels
    assert list(history["step"]) == [0]
    assert market.config.liquidity_backstop == "always"


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
    pd.testing.assert_frame_equal(market_a.get_debug_history(), market_b.get_debug_history())
    assert not market_a.get_event_history().equals(market_c.get_event_history())


def test_history_contains_summary_columns_only() -> None:
    market = Market(seed=7)
    market.gen(steps=5)

    history = market.get_history()

    expected = {
        "step",
        "day",
        "session_step",
        "session_phase",
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


def test_history_only_mode_preserves_summary_but_disables_event_and_debug_apis() -> None:
    full_market = Market(seed=13, config={"preset": "balanced", "logging_mode": "full"})
    compact_market = Market(seed=13, config={"preset": "balanced", "logging_mode": "history_only"})

    full_market.gen(steps=40)
    compact_market.gen(steps=40)

    pd.testing.assert_frame_equal(full_market.get_history(), compact_market.get_history())
    assert compact_market.get()["step"] == full_market.get()["step"]

    with pytest.raises(RuntimeError, match="logging_mode='full'"):
        compact_market.get_event_history()
    with pytest.raises(RuntimeError, match="logging_mode='full'"):
        compact_market.get_debug_history()
    with pytest.raises(RuntimeError, match="logging_mode='full'"):
        compact_market.plot_diagnostics()


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


@pytest.mark.parametrize(
    ("preset", "seed", "steps", "expected_history", "expected_events", "expected_debug"),
    [
        (
            "balanced",
            101,
            18,
            "7882a953e6748d16a553fdf5546f107be1362e70cf187c34d35ce6c490bd3c32",
            "b906ab484dd79001f3ce6eaf8c0226876ed40bc6da3ab25d4492c72b27dbbb6e",
            "46e1346f421983081996a00c0af88cad51171cfda0147645da7dcd7d9402437d",
        ),
        (
            "trend",
            202,
            18,
            "a271ec0347b445f46eb1d9e64ebb28399ad8317a5a34d305643bd3c1bca867ef",
            "9050e29b6460757d3f57a1a7bd117f8e512bda4fbcd39055a62abb7f45c5c694",
            "cde0c2eaa71d658c4cad41f947587142ab596d82a975245cff9287178bf02ab3",
        ),
        (
            "volatile",
            303,
            18,
            "a76beb72cc9767eaeba024ba774e0eed967b2423f696282e3e20f9f1fd7681b1",
            "8bfac05cc3d42d9055dd9473b9b4b6fb287984dee5596064c79579a4e15fc55a",
            "be2ab075c56a8bd1d326e987492ece866ac759b6d1a11ea559bde235e8990630",
        ),
    ],
)
def test_seeded_path_hashes_match_refactor_baseline(
    preset: str,
    seed: int,
    steps: int,
    expected_history: str,
    expected_events: str,
    expected_debug: str,
) -> None:
    market = Market(seed=seed, config={"preset": preset})
    market.gen(steps=steps)

    assert _stable_frame_hash(market.get_history()) == expected_history
    assert _stable_frame_hash(market.get_event_history()) == expected_events
    assert _stable_frame_hash(market.get_debug_history()) == expected_debug
