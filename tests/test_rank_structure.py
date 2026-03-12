from __future__ import annotations

from tests._realism_cache import realism_bundle


def test_rank_structure_retains_depth_gradient_and_variation() -> None:
    profiles, aggregate = realism_bundle()

    for side in (aggregate.bid_rank_mean_depth, aggregate.ask_rank_mean_depth):
        assert side[0] > side[4]
        assert side[1] > side[4]
        assert side[2] > side[4]
        assert (side[0] + side[1]) > (side[3] + side[4])

    for side in (aggregate.bid_rank_std_depth, aggregate.ask_rank_std_depth):
        assert side[0] > 0.5
        assert side[1] > 0.4
        assert side[2] > 0.3

    assert aggregate.bid_rank_mean_depth[0] > aggregate.bid_rank_mean_depth[2]
    assert aggregate.ask_rank_mean_depth[0] > aggregate.ask_rank_mean_depth[2]
    assert aggregate.bid_rank_std_depth[0] > aggregate.bid_rank_std_depth[4]
    assert aggregate.ask_rank_std_depth[0] > aggregate.ask_rank_std_depth[4]

    assert 0.15 <= aggregate.bid_gap_gt1_share_top5 <= 0.40
    assert 0.20 <= aggregate.ask_gap_gt1_share_top5 <= 0.40
    assert aggregate.bid_gap_run_median <= 5.0
    assert aggregate.ask_gap_run_median <= 5.0

    for profile in profiles:
        assert profile.near_touch_connectivity_bid >= 0.75
        assert profile.near_touch_connectivity_ask >= 0.72
