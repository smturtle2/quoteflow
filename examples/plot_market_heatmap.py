from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from orderwave import Market


@dataclass(frozen=True)
class DepthHeatmap:
    steps: np.ndarray
    ask_levels: int
    row_labels: list[str]
    signed_depth: np.ndarray


def simulate_market(
    *,
    steps: int,
    seed: int,
    preset: str,
    init_price: float,
    tick_size: float,
    levels: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    market = Market(
        init_price=init_price,
        tick_size=tick_size,
        levels=levels,
        seed=seed,
        config={"preset": preset},
    )

    snapshots = [market.get()]
    for _ in range(steps):
        snapshots.append(market.step())
    return market.get_history(), snapshots


def build_depth_heatmap(snapshots: list[dict[str, object]]) -> DepthHeatmap:
    steps = np.array([int(snapshot["step"]) for snapshot in snapshots], dtype=int)
    max_asks = max(1, max(len(snapshot["asks"]) for snapshot in snapshots))
    max_bids = max(1, max(len(snapshot["bids"]) for snapshot in snapshots))
    ask_labels = [f"ask {level}" for level in range(max_asks, 0, -1)]
    bid_labels = [f"bid {level}" for level in range(1, max_bids + 1)]
    row_labels = ask_labels + bid_labels
    signed_depth = np.zeros((len(row_labels), len(snapshots)), dtype=float)

    for column, snapshot in enumerate(snapshots):
        asks = list(snapshot["asks"])
        bids = list(snapshot["bids"])

        for depth_index, level in enumerate(asks, start=0):
            signed_depth[max_asks - depth_index - 1, column] = -float(level["qty"])

        bid_offset = max_asks
        for depth_index, level in enumerate(bids, start=0):
            signed_depth[bid_offset + depth_index, column] = float(level["qty"])

    return DepthHeatmap(
        steps=steps,
        ask_levels=max_asks,
        row_labels=row_labels,
        signed_depth=signed_depth,
    )


def plot_market_heatmap(
    history: pd.DataFrame,
    heatmap: DepthHeatmap,
    *,
    title: str,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba

    figure, (price_ax, strength_ax, heat_ax) = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True,
        height_ratios=(1.0, 0.8, 2.3),
        constrained_layout=True,
    )

    steps = history["step"].to_numpy()
    mid_price = history["mid_price"].to_numpy()
    last_price = history["last_price"].to_numpy()
    best_bid = history["best_bid"].to_numpy()
    best_ask = history["best_ask"].to_numpy()
    trade_strength = history["trade_strength"].to_numpy()

    price_ax.plot(steps, mid_price, color="#0f172a", linewidth=1.7, label="Mid price")
    price_ax.plot(steps, last_price, color="#f97316", linewidth=1.1, alpha=0.9, label="Last trade")
    price_ax.fill_between(
        steps,
        best_bid,
        best_ask,
        color="#94a3b8",
        alpha=0.22,
        label="Bid/ask spread",
    )
    price_ax.set_ylabel("Price")
    price_ax.set_title(title)
    price_ax.grid(alpha=0.25, linestyle="--")
    price_ax.legend(loc="upper left", ncol=3, frameon=False)

    strength_ax.axhline(0.0, color="#0f172a", linewidth=0.9, alpha=0.5)
    strength_ax.fill_between(
        steps,
        0.0,
        trade_strength,
        where=trade_strength >= 0.0,
        color="#38bdf8",
        alpha=0.35,
        interpolate=True,
    )
    strength_ax.fill_between(
        steps,
        0.0,
        trade_strength,
        where=trade_strength < 0.0,
        color="#ef4444",
        alpha=0.35,
        interpolate=True,
    )
    strength_ax.plot(steps, trade_strength, color="#0f172a", linewidth=1.0, label="Trade strength")
    strength_ax.set_ylabel("Strength")
    strength_ax.set_ylim(-1.05, 1.05)
    strength_ax.grid(alpha=0.2, linestyle="--")
    strength_ax.legend(loc="upper left", frameon=False)

    max_abs_depth = float(np.max(np.abs(heatmap.signed_depth)))
    if max_abs_depth <= 0.0:
        max_abs_depth = 1.0

    max_bin = max(1, int(np.ceil(max_abs_depth)))
    white = np.array(to_rgba("#ffffff"))
    black = np.array(to_rgba("#000000"))
    ask_red = np.array(to_rgba("#ef4444"))
    bid_blue = np.array(to_rgba("#38bdf8"))

    def blend(start: np.ndarray, end: np.ndarray, fraction: float) -> tuple[float, float, float, float]:
        return tuple(start + ((end - start) * fraction))

    neg_colors: list[tuple[float, float, float, float]] = []
    pos_colors: list[tuple[float, float, float, float]] = []
    for magnitude in range(max_bin, 0, -1):
        fraction = 0.0 if max_bin == 1 else (magnitude - 1) / (max_bin - 1)
        neg_colors.append(blend(white, ask_red, fraction))
    for magnitude in range(1, max_bin + 1):
        fraction = 0.0 if max_bin == 1 else (magnitude - 1) / (max_bin - 1)
        pos_colors.append(blend(white, bid_blue, fraction))

    heat_cmap = ListedColormap(neg_colors + [tuple(black)] + pos_colors, name="orderwave_signed_depth")
    heat_norm = BoundaryNorm(
        np.arange(-max_bin - 0.5, max_bin + 1.5, 1.0),
        heat_cmap.N,
    )

    x_edges = np.arange(len(heatmap.steps) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(heatmap.row_labels) + 1, dtype=float) - 0.5
    mesh = heat_ax.pcolormesh(
        x_edges,
        y_edges,
        heatmap.signed_depth,
        cmap=heat_cmap,
        norm=heat_norm,
        shading="flat",
        edgecolors=(1.0, 1.0, 1.0, 0.08),
        linewidth=0.05,
    )
    heat_ax.set_xlim(float(heatmap.steps[0]) - 0.5, float(heatmap.steps[-1]) + 0.5)
    heat_ax.set_ylim(len(heatmap.row_labels) - 0.5, -0.5)
    heat_ax.set_xlabel("Step")
    heat_ax.set_ylabel("Visible book level")
    heat_ax.set_yticks(np.arange(len(heatmap.row_labels), dtype=float))
    heat_ax.set_yticklabels(heatmap.row_labels)
    heat_ax.axhline(heatmap.ask_levels - 0.5, color="#0f172a", linewidth=0.8, alpha=0.35)
    heat_ax.grid(False)

    colorbar = figure.colorbar(mesh, ax=heat_ax, pad=0.015)
    colorbar.set_label("Visible depth (0 is black)")

    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot price history and visible order-book heatmap for an orderwave simulation."
    )
    parser.add_argument("--steps", type=int, default=1_000, help="Number of simulation steps to run.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--preset",
        choices=("balanced", "trend", "volatile"),
        default="balanced",
        help="Simulation preset.",
    )
    parser.add_argument("--levels", type=int, default=8, help="Visible depth levels to capture.")
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

    history, snapshots = simulate_market(
        steps=args.steps,
        seed=args.seed,
        preset=args.preset,
        init_price=args.init_price,
        tick_size=args.tick_size,
        levels=args.levels,
    )
    heatmap = build_depth_heatmap(snapshots)
    figure = plot_market_heatmap(
        history,
        heatmap,
        title=f"orderwave visible order-book heatmap ({args.preset}, seed={args.seed})",
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
