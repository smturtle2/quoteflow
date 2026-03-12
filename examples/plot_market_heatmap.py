from __future__ import annotations

import argparse
from pathlib import Path

from orderwave import Market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the built-in orderwave heatmap.")
    parser.add_argument("--steps", type=int, default=1_000, help="Number of simulation steps to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--levels", type=int, default=6, help="Visible depth for the runtime snapshot.")
    parser.add_argument("--init-price", type=float, default=100.0, help="Initial market price.")
    parser.add_argument("--tick-size", type=float, default=0.01, help="Tick size.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure dpi.")
    parser.add_argument(
        "--anchor",
        choices=("mid", "price"),
        default="mid",
        help="Heatmap anchor mode.",
    )
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
        capture="visual",
    )
    market.run(args.steps)
    if args.anchor == "price":
        figure = market.plot_heatmap(
            anchor="price",
            max_steps=min(args.steps, 1_200),
            price_window_ticks=16,
            title=f"orderwave fixed-level heatmap (seed={args.seed})",
        )
    else:
        figure = market.plot(
            max_steps=min(args.steps, 1_200),
            price_window_ticks=12,
            title=f"orderwave overview ({args.anchor}, seed={args.seed})",
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
