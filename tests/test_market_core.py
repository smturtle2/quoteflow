from __future__ import annotations

import pytest

from orderwave import Market, SimulationResult
from orderwave.history import SUMMARY_COLUMNS


def test_market_core_flow_and_history_shape() -> None:
    market = Market(seed=42)

    initial_snapshot = market.get()
    initial_history = market.get_history()

    assert initial_snapshot["step"] == 0
    assert list(initial_history.columns) == SUMMARY_COLUMNS
    assert len(initial_history) == 1

    stepped_snapshot = market.step()
    assert stepped_snapshot["step"] == 1

    generated_snapshot = market.gen(steps=4)
    assert generated_snapshot["step"] == 5

    result = market.run(steps=3)
    assert isinstance(result, SimulationResult)
    assert result.snapshot == market.get()
    assert result.history.equals(market.get_history())
    assert int(result.history.iloc[-1]["step"]) == 8


def test_market_config_mapping_and_validation() -> None:
    market = Market(seed=3, config={"market_rate": 3.0, "max_spread_ticks": 4})
    assert market.config.market_rate == 3.0
    assert market.config.max_spread_ticks == 4

    with pytest.raises(ValueError, match="mean_reversion"):
        Market(config={"mean_reversion": 1.5})

    with pytest.raises(ValueError, match="unknown MarketConfig fields"):
        Market(config={"preset": "trend"})
