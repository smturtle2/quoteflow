from __future__ import annotations

import numpy as np

from orderwave import Market
from orderwave._engine import _MarketEngine
from orderwave._model.scoring import score_limit_levels


def test_score_limit_levels_returns_normalized_probabilities() -> None:
    market = Market(seed=17, config={"preset": "balanced"})
    features = market._compute_features()

    levels, probs = score_limit_levels(
        "bid",
        "passive_lp",
        book=market._book,
        features=features,
        hidden_fair_tick=market._hidden_fair_tick,
        regime=market._regime,
        context=_MarketEngine(market)._current_context(),
        config=market.config,
        params=market._params,
    )

    assert len(levels) == len(probs)
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0)
    assert (-1 in levels) is True
