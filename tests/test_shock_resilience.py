from __future__ import annotations

from tests._realism_cache import realism_bundle


def test_shocks_create_asymmetric_cancel_and_refill_responses() -> None:
    _, aggregate = realism_bundle()

    assert aggregate.buy_shock_cancel_skew > 0.0
    assert aggregate.sell_shock_cancel_skew > 0.0
    assert aggregate.buy_shock_refill_skew < 0.0
    assert aggregate.sell_shock_refill_skew < 0.0


def test_impact_memory_persists_but_decays_across_horizons() -> None:
    _, aggregate = realism_bundle()

    first, second, fifth, tenth = aggregate.impact_decay_abs
    assert first > 0.05
    assert first > fifth
    assert first > tenth
    assert second < 0.20
