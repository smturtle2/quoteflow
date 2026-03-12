from __future__ import annotations

"""Profile generic microstructure realism for orderwave."""

from argparse import ArgumentParser
from json import dumps

from orderwave import Market
from orderwave._realism import aggregate_realism_profiles, profile_market_realism


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    args = parser.parse_args()

    profiles = []
    for seed in args.seeds:
        market = Market(seed=seed)
        profiles.append(profile_market_realism(market, steps=args.steps))

    payload = {
        "profiles": [profile.to_dict() for profile in profiles],
        "aggregate": aggregate_realism_profiles(profiles).to_dict(),
    }
    print(dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
