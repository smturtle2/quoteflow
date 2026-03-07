from __future__ import annotations

import argparse
from time import perf_counter

from orderwave import Market


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark orderwave simulation throughput.")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--preset", type=str, default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    market = Market(seed=args.seed, config={"preset": args.preset})

    started = perf_counter()
    market.gen(args.steps)
    elapsed = perf_counter() - started

    steps_per_second = args.steps / elapsed if elapsed > 0 else float("inf")
    snapshot = market.get()
    events = market.get_event_history()
    market_events = events.loc[events["event_type"] == "market"]
    market_buy_count = int((market_events["side"] == "buy").sum())
    market_sell_count = int((market_events["side"] == "sell").sum())
    total_market_events = market_buy_count + market_sell_count
    market_buy_share = market_buy_count / total_market_events if total_market_events > 0 else 0.0
    events_per_step = len(events) / max(args.steps, 1)

    print(f"preset={args.preset}")
    print(f"steps={args.steps}")
    print(f"elapsed_seconds={elapsed:.4f}")
    print(f"steps_per_second={steps_per_second:,.0f}")
    print(f"events_per_step={events_per_step:.2f}")
    print(
        "market_flow="
        f"buy={market_buy_count} "
        f"sell={market_sell_count} "
        f"buy_share={market_buy_share:.3f}"
    )
    print(
        "final="
        f"mid={snapshot['mid_price']:.4f} "
        f"spread={snapshot['spread']:.4f} "
        f"regime={snapshot['regime']}"
    )


if __name__ == "__main__":
    main()
