from __future__ import annotations

import math

from tests._realism_cache import realism_bundle


def test_realism_profile_stays_within_generic_microstructure_bounds() -> None:
    profiles, aggregate = realism_bundle()

    assert 0.60 <= aggregate.one_tick_spread_share <= 0.92
    assert 0.005 <= aggregate.wide_spread_share <= 0.08
    assert 0.0 <= aggregate.spread_acf1 <= 0.20
    assert -0.05 <= aggregate.trade_sign_acf1 <= 0.15
    assert aggregate.same_step_impact_corr >= 0.18
    assert -0.06 <= aggregate.next_step_impact_corr <= 0.08
    assert aggregate.visible_one_side_thin_share <= 0.03
    assert aggregate.visible_one_vs_many_share <= 0.03
    assert aggregate.full_book_extreme_share <= 0.12
    assert aggregate.avg_visible_levels_bid >= 4.0
    assert aggregate.avg_visible_levels_ask >= 4.0
    assert aggregate.avg_full_levels_bid >= 7.0
    assert aggregate.avg_full_levels_ask >= 7.0
    assert aggregate.near_touch_connectivity_bid >= 0.80
    assert aggregate.near_touch_connectivity_ask >= 0.80
    assert aggregate.visible_pair_entropy >= 0.45
    assert aggregate.full_pair_entropy >= 0.75
    assert math.isfinite(aggregate.spread_recovery_median)

    for profile in profiles:
        assert profile.visible_one_side_thin_share <= 0.04
        assert profile.visible_one_vs_many_share <= 0.04
        assert profile.full_book_extreme_share <= 0.15
        assert profile.avg_visible_levels_bid >= 4.0
        assert profile.avg_visible_levels_ask >= 4.0
        assert -0.08 <= profile.next_step_impact_corr <= 0.08
        assert math.isfinite(profile.spread_recovery_median)
