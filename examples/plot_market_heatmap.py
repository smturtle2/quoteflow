from __future__ import annotations

import argparse
from pathlib import Path

from orderwave import Market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the built-in orderwave overview plot.")
    parser.add_argument("--steps", type=int, default=1_000, help="Number of simulation steps to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--preset",
        choices=("balanced", "trend", "volatile"),
        default="balanced",
        help="Simulation preset.",
    )
    parser.add_argument("--levels", type=int, default=8, help="Visible depth rows to draw in the heatmap.")
    parser.add_argument("--init-price", type=float, default=100.0, help="Initial market price.")
    parser.add_argument("--tick-size", type=float, default=0.01, help="Tick size.")
    parser.add_argument("--dpi", type=int, default=140, help="Figure dpi.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, the figure is shown interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output is not None:
        import matplotlib

        matplotlib.use("Agg")

    market = Market(
        init_price=args.init_price,
        tick_size=args.tick_size,
        levels=args.levels,
        seed=args.seed,
        preset=args.preset,
    )
    market.run(args.steps)
    figure = market.plot(
        levels=args.levels,
        title=f"orderwave overview ({args.preset}, seed={args.seed})",
    )
    figure.set_dpi(args.dpi)

    if args.output is None:
        import matplotlib.pyplot as plt

        plt.show()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(f"saved plot to {args.output}")


if __name__ == "__main__":
    main()
