from __future__ import annotations

import math

from tests._realism_cache import realism_bundle


def test_realism_profile_stays_within_generic_microstructure_bounds() -> None:
    profiles, aggregate = realism_bundle()

    assert aggregate.normalized_net_drift <= 12.0
    assert 0.45 <= aggregate.variance_ratio_5 <= 1.10
    assert 0.25 <= aggregate.variance_ratio_20 <= 0.95
    assert 0.45 <= aggregate.one_tick_spread_share <= 0.80
    assert 0.01 <= aggregate.wide_spread_share <= 0.10
    assert -0.15 <= aggregate.spread_acf1 <= 0.25
    assert -0.05 <= aggregate.trade_sign_acf1 <= 0.10
    assert -0.05 <= aggregate.trade_sign_acf5 <= 0.08
    assert aggregate.same_step_impact_corr >= 0.35
    assert -0.10 <= aggregate.next_step_impact_corr <= 0.02
    assert aggregate.visible_one_side_thin_share <= 0.01
    assert aggregate.visible_one_vs_many_share <= 0.01
    assert aggregate.full_book_extreme_share <= 0.01
    assert aggregate.avg_visible_levels_bid >= 4.8
    assert aggregate.avg_visible_levels_ask >= 4.8
    assert aggregate.avg_full_levels_bid >= 15.0
    assert aggregate.avg_full_levels_ask >= 15.0
    assert aggregate.near_touch_connectivity_bid >= 0.75
    assert aggregate.near_touch_connectivity_ask >= 0.72
    assert aggregate.bid_gap_gt1_share_top5 >= 0.18
    assert aggregate.ask_gap_gt1_share_top5 >= 0.20
    assert aggregate.full_pair_entropy >= 0.85
    assert math.isfinite(aggregate.spread_recovery_median)

    for profile in profiles:
        assert profile.normalized_net_drift <= 12.0
        assert 0.20 <= profile.variance_ratio_20 <= 1.05
        assert profile.visible_one_side_thin_share <= 0.01
        assert profile.visible_one_vs_many_share <= 0.01
        assert profile.full_book_extreme_share <= 0.01
        assert profile.wide_spread_share <= 0.10
        assert -0.10 <= profile.trade_sign_acf1 <= 0.12
        assert math.isfinite(profile.spread_recovery_median)
