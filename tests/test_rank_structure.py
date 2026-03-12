from __future__ import annotations

from tests._realism_cache import realism_bundle


def test_rank_structure_retains_depth_gradient_and_variation() -> None:
    profiles, aggregate = realism_bundle()

    assert aggregate.connected_depth_share > aggregate.isolated_depth_share

    for side in (aggregate.bid_rank_mean_depth, aggregate.ask_rank_mean_depth):
        assert side[0] > side[1] > side[2]
        assert side[2] > side[3] > side[4]

    for side in (aggregate.bid_rank_std_depth, aggregate.ask_rank_std_depth):
        assert side[0] > 1.5
        assert side[1] > 1.0
        assert side[2] > 0.8

    bid_rank2 = [profile.bid_rank_mean_depth[1] for profile in profiles]
    ask_rank2 = [profile.ask_rank_mean_depth[1] for profile in profiles]
    bid_rank5 = [profile.bid_rank_mean_depth[4] for profile in profiles]
    ask_rank5 = [profile.ask_rank_mean_depth[4] for profile in profiles]

    assert max(bid_rank2) - min(bid_rank2) > 0.10
    assert max(ask_rank2) - min(ask_rank2) > 0.10
    assert max(bid_rank5) - min(bid_rank5) > 0.05
    assert max(ask_rank5) - min(ask_rank5) > 0.05
