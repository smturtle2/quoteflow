from __future__ import annotations

from tests._realism_cache import realism_bundle


def test_rank_structure_retains_depth_gradient_and_variation() -> None:
    profiles, aggregate = realism_bundle()

    for side in (aggregate.bid_rank_mean_depth, aggregate.ask_rank_mean_depth):
        assert side[0] > side[1] > side[2]
        assert side[2] >= side[3] >= side[4]

    for side in (aggregate.bid_rank_std_depth, aggregate.ask_rank_std_depth):
        assert side[0] > 0.5
        assert side[1] > 0.3
        assert side[2] > 0.2

    bid_rank1 = [profile.bid_rank_mean_depth[0] for profile in profiles]
    ask_rank1 = [profile.ask_rank_mean_depth[0] for profile in profiles]
    assert max(bid_rank1) - min(bid_rank1) > 0.05
    assert max(ask_rank1) - min(ask_rank1) > 0.05

    assert aggregate.bid_gap_gt1_share_top5 <= 0.20
    assert aggregate.ask_gap_gt1_share_top5 <= 0.25
    assert aggregate.bid_gap_run_median <= 2.0
    assert aggregate.ask_gap_run_median <= 2.0
