from __future__ import annotations

from functools import lru_cache

from orderwave import Market
from orderwave._realism import RealismProfile, aggregate_realism_profiles, profile_market_realism


@lru_cache(maxsize=4)
def realism_bundle(*, steps: int = 5_000, seeds: tuple[int, ...] = (11, 17, 23)) -> tuple[tuple[RealismProfile, ...], RealismProfile]:
    profiles = tuple(profile_market_realism(Market(seed=seed), steps=steps) for seed in seeds)
    return profiles, aggregate_realism_profiles(profiles)
