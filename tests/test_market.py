from __future__ import annotations

import pandas as pd
import pytest

from orderwave import Market
from orderwave._model.samplers import sample_participant_events
from orderwave.book import OrderBook
from orderwave.metrics import compute_features
from orderwave.validation import _stable_frame_hash


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
        ),
        bid_levels=market._book.top_levels("bid", market.levels),
        ask_levels=market._book.top_levels("ask", market.levels),
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
        ),
        bid_levels=market._book.top_levels("bid", market.levels),
        ask_levels=market._book.top_levels("ask", market.levels),
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


@pytest.mark.parametrize(
    ("preset", "seed", "steps", "expected_history", "expected_events", "expected_debug"),
    [
        (
            "balanced",
            101,
            18,
            "7882a953e6748d16a553fdf5546f107be1362e70cf187c34d35ce6c490bd3c32",
            "91a20071cfdbd582953914390f25aae54b0d662b8f83bcd9861bcf58f1975ec1",
            "e1079296740056a117063a02178600aa4ef6e8c391ee5f1a788025a42c8bba7e",
        ),
        (
            "trend",
            202,
            18,
            "a271ec0347b445f46eb1d9e64ebb28399ad8317a5a34d305643bd3c1bca867ef",
            "ca981134881738be5cf9b8b5aa40a16d2f3b37817cdfee791d8dc228f5644559",
            "d6007ca86452c1cf083694c35cc0adcae7b8c494ee7fbbb6ff7ebf978cf7709a",
        ),
        (
            "volatile",
            303,
            18,
            "a76beb72cc9767eaeba024ba774e0eed967b2423f696282e3e20f9f1fd7681b1",
            "3b0455a23a91fe910fa2e633068b45b182905c24542597055d046c968829707c",
            "a36b14091cf487316350b916dbdeec491e942003c73c22cd1362772c93a0cdb7",
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


def test_step_pipeline_phases_preserve_mutation_boundaries() -> None:
    market = Market(seed=42)
    initial_step = market.get()["step"]
    initial_history = market.get_history().copy()
    previous_features = market._compute_features()

    step_state = market.advance_latent_state(previous_features)
    assert market.get()["step"] == initial_step
    pd.testing.assert_frame_equal(market.get_history(), initial_history)

    sampled_events = market.sample_step_events(step_state)
    assert market.get()["step"] == initial_step
    pd.testing.assert_frame_equal(market.get_history(), initial_history)

    step_outcome = market.apply_step_events(sampled_events)
    assert step_outcome.sampled_event_count == len(sampled_events)
    assert step_outcome.applied_event_count <= step_outcome.sampled_event_count
    assert market.get()["step"] == initial_step

    market.finalize_step(step_state, step_outcome)
    assert market.get()["step"] == initial_step + 1
    assert len(market.get_history()) == len(initial_history) + 1


def test_sample_step_events_is_deterministic_before_book_application() -> None:
    market_a = Market(seed=99, config={"preset": "trend"})
    market_b = Market(seed=99, config={"preset": "trend"})

    features_a = market_a._compute_features()
    features_b = market_b._compute_features()
    state_a = market_a.advance_latent_state(features_a)
    state_b = market_b.advance_latent_state(features_b)

    events_a = market_a.sample_step_events(state_a)
    events_b = market_b.sample_step_events(state_b)

    assert events_a == events_b


def test_canonical_budget_sampler_returns_valid_event_shapes() -> None:
    market = Market(seed=17, config={"preset": "balanced"})
    features = market._compute_features()
    step_state = market.advance_latent_state(features)

    events = sample_participant_events(
        book=market._book,
        features=step_state.previous_features,
        hidden_fair_tick=step_state.hidden_fair_tick,
        regime=step_state.regime,
        context=step_state.context,
        rng=market._rng,
        config=market.config,
        params=market._params,
    )

    assert all(event["type"] in {"limit", "market", "cancel"} for event in events)
    assert all(event["participant_type"] in {"passive_lp", "inventory_mm", "noise_taker", "informed_meta"} for event in events)
    assert all(event["qty"] > 0 for event in events)


def test_liquidity_backstop_always_restores_visible_depth() -> None:
    market = Market(seed=5, config={"liquidity_backstop": "always"})
    market._book = OrderBook(tick_size=market.tick_size)
    market._book.set_level("bid", 10000, 2)
    market._book.set_level("ask", 10001, 2)

    market._ensure_liquidity()

    assert len(market._book.top_levels("bid", market._minimum_visible_levels)) == market._minimum_visible_levels
    assert len(market._book.top_levels("ask", market._minimum_visible_levels)) == market._minimum_visible_levels


def test_liquidity_backstop_on_empty_only_restores_missing_side() -> None:
    market = Market(seed=6, config={"liquidity_backstop": "on_empty"})
    market._book = OrderBook(tick_size=market.tick_size)
    market._book.set_level("bid", 10000, 2)
    market._book.set_level("ask", 10001, 2)

    market._ensure_liquidity()
    assert len(market._book.top_levels("bid", market._minimum_visible_levels)) == 1
    assert len(market._book.top_levels("ask", market._minimum_visible_levels)) == 1

    market._book.cancel_level("ask", 10001, 2)
    market._ensure_liquidity()
    assert market._book.best_ask_tick is not None
    assert len(market._book.top_levels("bid", market._minimum_visible_levels)) == 1


def test_liquidity_backstop_off_leaves_empty_side_empty() -> None:
    market = Market(seed=7, config={"liquidity_backstop": "off"})
    market._book = OrderBook(tick_size=market.tick_size)
    market._book.set_level("bid", 10000, 2)
    market._book.set_level("ask", 10001, 2)
    market._book.cancel_level("ask", 10001, 2)

    market._ensure_liquidity()

    assert market._book.best_ask_tick is None
