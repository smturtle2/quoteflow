from __future__ import annotations

from tests._realism_cache import realism_bundle


def test_impact_memory_persists_but_decays_across_horizons() -> None:
    _, aggregate = realism_bundle()

    first, second, fifth, tenth = aggregate.impact_decay_abs
    short_horizon_peak = max(first, second)
    assert short_horizon_peak > 0.01
    assert ((first + second) / 2.0) >= ((fifth + tenth) / 2.0)


def test_random_state_space_is_not_collapsed_to_one_pair() -> None:
    _, aggregate = realism_bundle()

    assert aggregate.full_pair_entropy >= 0.85
    assert aggregate.near_touch_connectivity_bid >= 0.75
    assert aggregate.near_touch_connectivity_ask >= 0.72
    assert aggregate.bid_rank_std_depth[0] > 0.8
    assert aggregate.ask_rank_std_depth[0] > 0.8
