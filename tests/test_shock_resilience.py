from __future__ import annotations

from tests._realism_cache import realism_bundle


def test_impact_memory_persists_but_decays_across_horizons() -> None:
    _, aggregate = realism_bundle()

    first, second, fifth, tenth = aggregate.impact_decay_abs
    short_horizon_peak = max(first, second)
    assert short_horizon_peak > 0.02
    assert short_horizon_peak >= fifth
    assert short_horizon_peak >= tenth
    assert fifth >= tenth


def test_random_state_space_is_not_collapsed_to_one_pair() -> None:
    _, aggregate = realism_bundle()

    assert aggregate.visible_pair_entropy >= 0.20
    assert aggregate.full_pair_entropy >= 0.20
