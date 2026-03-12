from __future__ import annotations

import math

from tests._realism_cache import realism_bundle


def test_realism_profile_stays_within_generic_microstructure_bounds() -> None:
    profiles, aggregate = realism_bundle()

    assert aggregate.normalized_net_drift <= 4.0
    assert 0.20 <= aggregate.up_step_share <= 0.30
    assert 0.20 <= aggregate.down_step_share <= 0.30
    assert abs(aggregate.up_step_share - aggregate.down_step_share) <= 0.04
    assert 0.75 <= aggregate.variance_ratio_5 <= 1.15
    assert 0.65 <= aggregate.variance_ratio_20 <= 1.05
    assert 0.55 <= aggregate.one_tick_spread_share <= 0.78
    assert 0.01 <= aggregate.wide_spread_share <= 0.05
    assert -0.05 <= aggregate.spread_acf1 <= 0.20
    assert -0.10 <= aggregate.trade_sign_acf1 <= 0.06
    assert -0.03 <= aggregate.trade_sign_acf5 <= 0.05
    assert aggregate.same_step_impact_corr >= 0.35
    assert -0.06 <= aggregate.next_step_impact_corr <= 0.03
    assert aggregate.flow_return_sign_agreement >= 0.72
    assert aggregate.visible_one_side_thin_share <= 0.01
    assert aggregate.visible_one_vs_many_share <= 0.01
    assert aggregate.full_book_extreme_share <= 0.01
    assert aggregate.avg_visible_levels_bid >= 4.8
    assert aggregate.avg_visible_levels_ask >= 4.8
    assert aggregate.avg_full_levels_bid >= 10.5
    assert aggregate.avg_full_levels_ask >= 10.5
    assert aggregate.near_touch_connectivity_bid >= 0.74
    assert aggregate.near_touch_connectivity_ask >= 0.74
    assert 0.25 <= aggregate.bid_gap_gt1_share_top5 <= 0.45
    assert 0.25 <= aggregate.ask_gap_gt1_share_top5 <= 0.45
    assert aggregate.full_pair_entropy >= 0.85
    assert math.isfinite(aggregate.spread_recovery_median)

    for profile in profiles:
        assert profile.normalized_net_drift <= 5.0
        assert 0.18 <= profile.up_step_share <= 0.30
        assert 0.18 <= profile.down_step_share <= 0.30
        assert abs(profile.up_step_share - profile.down_step_share) <= 0.06
        assert 0.55 <= profile.variance_ratio_20 <= 1.10
        assert profile.visible_one_side_thin_share <= 0.01
        assert profile.visible_one_vs_many_share <= 0.01
        assert profile.full_book_extreme_share <= 0.01
        assert profile.wide_spread_share <= 0.06
        assert -0.12 <= profile.trade_sign_acf1 <= 0.08
        assert profile.flow_return_sign_agreement >= 0.68
        assert math.isfinite(profile.spread_recovery_median)
