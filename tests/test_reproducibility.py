from __future__ import annotations

import pandas as pd

from orderwave import Market


def test_same_seed_produces_same_history() -> None:
    market_a = Market(seed=11)
    market_b = Market(seed=11)

    history_a = market_a.run(steps=200).history
    history_b = market_b.run(steps=200).history

    pd.testing.assert_frame_equal(history_a, history_b)
    assert market_a.get() == market_b.get()


def test_different_seed_changes_path() -> None:
    market_a = Market(seed=11)
    market_b = Market(seed=12)

    history_a = market_a.run(steps=200).history
    history_b = market_b.run(steps=200).history

    assert not history_a.equals(history_b)
