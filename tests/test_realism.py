from __future__ import annotations

import math

from orderwave import Market
from orderwave._realism import aggregate_realism_profiles, profile_market_realism


def test_realism_profile_stays_within_generic_microstructure_bounds() -> None:
    profiles = [profile_market_realism(Market(seed=seed), steps=5_000) for seed in (11, 17, 23)]
    aggregate = aggregate_realism_profiles(profiles)

    assert 0.40 <= aggregate.one_tick_spread_share <= 0.90
    assert 0.01 <= aggregate.wide_spread_share <= 0.25
    assert 0.30 <= aggregate.spread_acf1 <= 0.90
    assert 0.05 <= aggregate.trade_sign_acf1 <= 0.25
    assert aggregate.same_step_impact_corr >= 0.15
    assert aggregate.next_step_impact_corr >= 0.02
    assert aggregate.bid_gap_gt1_share_top5 >= 0.03
    assert aggregate.ask_gap_gt1_share_top5 >= 0.03
    assert 1.0 <= aggregate.spread_recovery_median <= 15.0

    for profile in profiles:
        assert 0.0 <= profile.one_tick_spread_share <= 1.0
        assert profile.same_step_impact_corr >= 0.15
        assert profile.next_step_impact_corr >= 0.02
        assert profile.wide_spread_share < 0.30
        assert profile.bid_gap_gt1_share_top5 >= 0.025
        assert profile.ask_gap_gt1_share_top5 >= 0.025
        assert math.isfinite(profile.spread_recovery_median)
