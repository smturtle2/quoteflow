from __future__ import annotations

import pandas as pd

from orderwave import Market
from orderwave._engine import _MarketEngine


def test_engine_advance_and_sample_do_not_mutate_public_state() -> None:
    market = Market(seed=42)
    engine = _MarketEngine(market)

    initial_snapshot = market.get()
    initial_history = market.get_history().copy()

    step_state = engine._advance_latent_state(market._compute_features())
    sampled_events = engine._sample_step_events(step_state)

    assert market.get() == initial_snapshot
    pd.testing.assert_frame_equal(market.get_history(), initial_history)
    assert isinstance(sampled_events, list)


def test_engine_apply_and_finalize_advance_history_once() -> None:
    market = Market(seed=17, config={"preset": "balanced"})
    engine = _MarketEngine(market)

    step_state = engine._advance_latent_state(market._compute_features())
    sampled_events = engine._sample_step_events(step_state)
    step_outcome = engine._apply_step_events(sampled_events)

    assert step_outcome.sampled_event_count == len(sampled_events)
    assert step_outcome.applied_event_count <= step_outcome.sampled_event_count
    assert market.get()["step"] == 0

    engine._finalize_step(step_state, step_outcome)

    assert market.get()["step"] == 1
    assert list(market.get_history()["step"]) == [0, 1]
