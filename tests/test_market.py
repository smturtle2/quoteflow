from __future__ import annotations

import hashlib
import importlib
import json
import math

import pandas as pd
import pytest

from orderwave import Market
from orderwave.market import BookLevel, MarketSnapshot, SimulationResult


def _stable_frame_hash(frame: pd.DataFrame) -> str:
    def normalize(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): normalize(inner) for key, inner in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if value is pd.NA:
            return None
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return normalize(value.item())
            except Exception:
                pass
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return str(value)
            return round(value, 12)
        return value

    payload = {
        "columns": list(frame.columns),
        "records": [normalize(record) for record in frame.to_dict(orient="records")],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
    assert market.config.liquidity_backstop == "on_empty"


def test_get_snapshot_exposes_typed_view_and_round_trips_to_dict() -> None:
    market = Market(seed=42, preset="trend")

    snapshot = market.get_snapshot()

    assert isinstance(snapshot, MarketSnapshot)
    assert snapshot.to_dict() == market.get()
    assert all(isinstance(level, BookLevel) for level in snapshot.bids)
    assert all(isinstance(level, BookLevel) for level in snapshot.asks)


def test_market_hides_internal_engine_stage_methods() -> None:
    market_attrs = set(dir(Market))
    market = Market(seed=1)

    assert "advance_latent_state" not in market_attrs
    assert "sample_step_events" not in market_attrs
    assert "apply_step_events" not in market_attrs
    assert "finalize_step" not in market_attrs
    assert hasattr(market, "_engine") is False


def test_orderwave_model_stub_rejects_internal_symbol_access() -> None:
    module = importlib.import_module("orderwave.model")

    with pytest.raises(AttributeError, match="orderwave.model is internal"):
        getattr(module, "sample_participant_events")


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


def test_public_constructor_kwargs_override_nested_config() -> None:
    market = Market(
        seed=7,
        config={"preset": "balanced", "logging_mode": "full", "liquidity_backstop": "always"},
        preset="trend",
        logging_mode="history_only",
        liquidity_backstop="off",
    )

    assert market.config.preset == "trend"
    assert market.config.logging_mode == "history_only"
    assert market.config.liquidity_backstop == "off"


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
        "visible_levels_bid",
        "visible_levels_ask",
        "drought_age",
        "recovery_pressure",
        "impact_residue",
        "regime_dwell",
        "inventory_pressure",
    }
    assert set(history.columns) == expected
    assert "bids" not in history.columns
    assert "asks" not in history.columns


def test_history_only_mode_preserves_summary_but_disables_event_and_debug_apis() -> None:
    full_market = Market(seed=13, preset="balanced", logging_mode="full")
    compact_market = Market(seed=13, preset="balanced", logging_mode="history_only")

    full_market.gen(steps=40)
    compact_result = compact_market.run(steps=40)

    pd.testing.assert_frame_equal(full_market.get_history(), compact_market.get_history())
    assert compact_market.get()["step"] == full_market.get()["step"]
    assert isinstance(compact_result, SimulationResult)
    assert compact_result.event_history is None
    assert compact_result.debug_history is None
    assert compact_result.labeled_event_history is None

    with pytest.raises(RuntimeError, match="logging_mode='full'"):
        compact_market.get_event_history()
    with pytest.raises(RuntimeError, match="logging_mode='full'"):
        compact_market.get_debug_history()
    with pytest.raises(RuntimeError, match="logging_mode='full'"):
        compact_market.plot_diagnostics()


def test_run_returns_bundle_with_labeled_event_history() -> None:
    market = Market(seed=21, preset="volatile")

    result = market.run(steps=35)

    assert isinstance(result, SimulationResult)
    assert result.snapshot.to_dict() == market.get()
    pd.testing.assert_frame_equal(result.history, market.get_history())
    assert result.event_history is not None
    assert result.debug_history is not None
    assert result.labeled_event_history is not None
    assert list(result.labeled_event_history.columns[-19:]) == [
        "microphase",
        "source",
        "participant_type",
        "meta_order_id",
        "meta_order_side",
        "meta_order_progress",
        "burst_state",
        "shock_state",
        "drought_age",
        "recovery_pressure",
        "impact_residue",
        "regime_dwell",
        "inventory_pressure",
        "flow_toxicity",
        "maker_stress",
        "quote_revision_wave",
        "refill_pressure",
        "visible_levels_bid",
        "visible_levels_ask",
    ]


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
            "95cb82d2cbab0854de2827d4341e4e9f62dc95e655736b36b9c7b320fb218228",
            "915ad5a426ff70d805e5c869d5aee50e03761db7a253d8d04bba8039f1a1284d",
            "e994f8a6b72a9425b646684f276437e71d5195dccd9cb3f56793473477c7b43f",
        ),
        (
            "trend",
            202,
            18,
            "c816abba07c77d9ed7f477b7d555870945a8e1e7cf2bcc76e3cea6a293f4d1b3",
            "0e9014a8ce89f4c4edc7cb4963954c5c387002ad643fcd442493f6c4fa43da97",
            "faa2bd20bbeff656d4168254246296b832e3ed7694a98cb0011b57bb5b133f52",
        ),
        (
            "volatile",
            303,
            18,
            "17ed2713f9fa9031f1294ec4a788a6631ced1d87293b3b435f08b5da0b30f382",
            "0bc968bc758d09c9269e3d88ee1291020b0ace8694da0b085c656213651bce17",
            "8a28eb06d8fc46fa0332940c58c8d27df618b6fb3add443f479a213d9a8e37b8",
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
