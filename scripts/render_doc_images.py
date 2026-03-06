from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    init_price: float = 100.0,
    tick_size: float = 0.01,
    levels: int = 8,
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
        for depth_index, level in enumerate(snapshot["asks"]):
            signed_depth[max_asks - depth_index - 1, column] = -float(level["qty"])

        bid_offset = max_asks
        for depth_index, level in enumerate(snapshot["bids"]):
            signed_depth[bid_offset + depth_index, column] = float(level["qty"])

    return DepthHeatmap(
        steps=steps,
        ask_levels=max_asks,
        row_labels=row_labels,
        signed_depth=signed_depth,
    )


def setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 10.5,
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.35,
            "savefig.facecolor": "white",
        }
    )


def signed_depth_style(max_abs_depth: float):
    from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba

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

    cmap = ListedColormap(neg_colors + [tuple(black)] + pos_colors, name="orderwave_signed_depth")
    norm = BoundaryNorm(np.arange(-max_bin - 0.5, max_bin + 1.5, 1.0), cmap.N)
    return cmap, norm


def render_overview(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    history, snapshots = simulate_market(steps=720, seed=17, preset="trend", levels=8)
    heatmap = build_depth_heatmap(snapshots)
    steps = history["step"].to_numpy()

    figure, (price_ax, strength_ax, heat_ax) = plt.subplots(
        3,
        1,
        figsize=(15, 10.5),
        sharex=True,
        height_ratios=(1.05, 0.8, 2.3),
        constrained_layout=True,
    )

    mid_price = history["mid_price"].to_numpy()
    last_price = history["last_price"].to_numpy()
    best_bid = history["best_bid"].to_numpy()
    best_ask = history["best_ask"].to_numpy()
    trade_strength = history["trade_strength"].to_numpy()

    price_ax.plot(steps, mid_price, color="#0f172a", linewidth=1.9, label="Mid price")
    price_ax.plot(steps, last_price, color="#f97316", linewidth=1.25, alpha=0.95, label="Last trade")
    price_ax.fill_between(steps, best_bid, best_ask, color="#94a3b8", alpha=0.2, label="Spread")
    price_ax.grid(linestyle="--")
    price_ax.legend(loc="upper left", frameon=False, ncol=3)
    price_ax.set_ylabel("Price")
    price_ax.set_title("Order flow becomes price")
    price_ax.text(
        0.99,
        0.04,
        f"preset=trend  seed=17  steps={int(steps[-1])}",
        transform=price_ax.transAxes,
        ha="right",
        va="bottom",
        color="#475569",
    )

    strength_ax.axhline(0.0, color="#0f172a", linewidth=0.9, alpha=0.45)
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
    strength_ax.plot(steps, trade_strength, color="#0f172a", linewidth=1.0)
    strength_ax.set_ylabel("Trade strength")
    strength_ax.set_ylim(-1.05, 1.05)
    strength_ax.grid(linestyle="--")

    max_abs_depth = max(1.0, float(np.max(np.abs(heatmap.signed_depth))))
    max_bin = max(1, int(np.ceil(max_abs_depth)))
    cmap, norm = signed_depth_style(max_abs_depth)
    x_edges = np.arange(len(heatmap.steps) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(heatmap.row_labels) + 1, dtype=float) - 0.5
    mesh = heat_ax.pcolormesh(
        x_edges,
        y_edges,
        heatmap.signed_depth,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors=(1.0, 1.0, 1.0, 0.07),
        linewidth=0.04,
    )
    heat_ax.set_xlim(float(heatmap.steps[0]) - 0.5, float(heatmap.steps[-1]) + 0.5)
    heat_ax.set_ylim(len(heatmap.row_labels) - 0.5, -0.5)
    heat_ax.set_xlabel("Step")
    heat_ax.set_ylabel("Visible book level")
    heat_ax.set_yticks(np.arange(len(heatmap.row_labels), dtype=float))
    heat_ax.set_yticklabels(heatmap.row_labels)
    heat_ax.axhline(heatmap.ask_levels - 0.5, color="#0f172a", linewidth=0.85, alpha=0.35)
    colorbar = figure.colorbar(mesh, ax=heat_ax, pad=0.015)
    colorbar.set_label("Signed visible depth")
    tick_values = sorted(set([-max_bin, -(max_bin // 2), 0, max_bin // 2, max_bin]))
    colorbar.set_ticks(tick_values)

    figure.savefig(outdir / "orderwave-overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_presets(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    presets = [
        ("balanced", "#0f766e"),
        ("trend", "#f97316"),
        ("volatile", "#dc2626"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    figure.suptitle("Preset behaviors at a glance", fontsize=16, fontweight="bold")
    legend_handles = None
    legend_labels = None

    for axis, (preset, color) in zip(axes, presets):
        history, _ = simulate_market(steps=480, seed=42, preset=preset, levels=6)
        steps = history["step"].to_numpy()
        mid = history["mid_price"].to_numpy()
        last = history["last_price"].to_numpy()
        spread = history["spread"].to_numpy()
        mid_ret = history["mid_price"].diff().fillna(0.0)

        axis.plot(steps, mid, color=color, linewidth=1.8, label="Mid")
        axis.plot(steps, last, color="#0f172a", linewidth=0.9, alpha=0.75, label="Last")
        axis.fill_between(steps, mid - (spread / 2.0), mid + (spread / 2.0), color=color, alpha=0.12)
        axis.set_title(preset.capitalize())
        axis.grid(linestyle="--")
        axis.set_xlabel("Step")
        if axis is axes[0]:
            axis.set_ylabel("Price")

        stats = (
            f"mean spread: {spread.mean():.3f}\n"
            f"ret std: {mid_ret.std():.3f}\n"
            f"range: {mid.max() - mid.min():.3f}"
        )
        axis.text(
            0.03,
            0.97,
            stats,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#e2e8f0", "boxstyle": "round,pad=0.35"},
        )
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()

    if legend_handles is not None and legend_labels is not None:
        figure.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)
    figure.savefig(outdir / "orderwave-presets.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_diagnostics(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    history, _ = simulate_market(steps=4_000, seed=9, preset="balanced", levels=6)
    mid_ret = history["mid_price"].diff().fillna(0.0)
    next_ret = mid_ret.shift(-1).fillna(0.0)
    abs_ret = mid_ret.abs()
    spread = history["spread"]
    imbalance = history["depth_imbalance"].clip(-1.0, 1.0)
    bins = np.linspace(-1.0, 1.0, 9)
    bin_index = np.digitize(imbalance, bins[1:-1], right=False)
    binned_signal = pd.DataFrame({"bin": bin_index, "next_ret": next_ret}).groupby("bin")["next_ret"].mean()
    x_positions = np.arange(len(bins) - 1)
    y_values = np.array([float(binned_signal.get(index, 0.0)) for index in x_positions])
    x_labels = [f"{bins[index]:.2f}\n{bins[index + 1]:.2f}" for index in x_positions]
    acf = np.array([abs_ret.autocorr(lag=lag) for lag in range(1, 13)])
    acf = np.nan_to_num(acf, nan=0.0)
    regime_share = history["regime"].value_counts(normalize=True).reindex(
        ["calm", "directional", "stressed"],
        fill_value=0.0,
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 8.4), constrained_layout=True)
    figure.suptitle("Microstructure diagnostics", fontsize=16, fontweight="bold")

    axes[0, 0].hist(spread, bins=18, color="#0f766e", alpha=0.85, edgecolor="white")
    axes[0, 0].set_title("Spread distribution")
    axes[0, 0].set_xlabel("Spread")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(linestyle="--")

    axes[0, 1].plot(x_positions, y_values, color="#2563eb", linewidth=1.8, marker="o")
    axes[0, 1].axhline(0.0, color="#0f172a", linewidth=0.9, alpha=0.4)
    axes[0, 1].set_title("Depth imbalance -> next mid return")
    axes[0, 1].set_xlabel("Imbalance bin")
    axes[0, 1].set_ylabel("Mean next return")
    axes[0, 1].set_xticks(x_positions)
    axes[0, 1].set_xticklabels(x_labels)
    axes[0, 1].grid(linestyle="--")

    axes[1, 0].bar(np.arange(1, 13), acf, color="#f97316", width=0.7)
    axes[1, 0].axhline(0.0, color="#0f172a", linewidth=0.9, alpha=0.4)
    axes[1, 0].set_title("Absolute return autocorrelation")
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].set_ylabel("Autocorr")
    axes[1, 0].grid(linestyle="--")

    axes[1, 1].bar(
        regime_share.index,
        regime_share.values,
        color=["#0f766e", "#f97316", "#dc2626"],
        width=0.65,
    )
    axes[1, 1].set_title("Regime occupancy")
    axes[1, 1].set_ylabel("Share")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(axis="y", linestyle="--")

    figure.savefig(outdir / "orderwave-diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render documentation images for orderwave.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "assets",
        help="Output directory for generated documentation images.",
    )
    return parser.parse_args()


def main() -> None:
    setup_matplotlib()
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    render_overview(args.outdir)
    render_presets(args.outdir)
    render_diagnostics(args.outdir)
    print(f"rendered documentation images to {args.outdir}")


if __name__ == "__main__":
    main()
