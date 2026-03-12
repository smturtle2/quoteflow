from __future__ import annotations

import math

from tests._realism_cache import realism_bundle


def test_realism_profile_stays_within_generic_microstructure_bounds() -> None:
    profiles, aggregate = realism_bundle()

    assert 0.72 <= aggregate.one_tick_spread_share <= 0.90
    assert 0.05 <= aggregate.wide_spread_share <= 0.18
    assert 0.0 <= aggregate.spread_acf1 <= 0.15
    assert 0.15 <= aggregate.trade_sign_acf1 <= 0.32
    assert aggregate.same_step_impact_corr >= 0.45
    assert aggregate.next_step_impact_corr >= 0.05
    assert 0.15 <= aggregate.bid_gap_gt1_share_top5 <= 0.40
    assert 0.15 <= aggregate.ask_gap_gt1_share_top5 <= 0.40
    assert 1.0 <= aggregate.spread_recovery_median <= 4.0
    assert 0.05 <= aggregate.tight_regime_share <= 0.40
    assert 0.05 <= aggregate.fragile_regime_share <= 0.40

    for profile in profiles:
        assert 0.0 <= profile.one_tick_spread_share <= 1.0
        assert profile.same_step_impact_corr >= 0.40
        assert profile.next_step_impact_corr > 0.0
        assert profile.trade_sign_acf1 >= 0.15
        assert profile.bid_gap_gt1_share_top5 < 0.45
        assert profile.ask_gap_gt1_share_top5 < 0.45
        assert math.isfinite(profile.spread_recovery_median)
