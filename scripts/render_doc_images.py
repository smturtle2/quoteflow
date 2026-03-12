from __future__ import annotations

"""Regenerate documentation images for the current orderwave API."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

from orderwave import Market


def render_overview(outdir: Path) -> None:
    market = Market(seed=42, capture="visual")
    market.run(steps=720)
    figure = market.plot(
        max_steps=720,
        price_window_ticks=12,
        title="Regime-aware order flow becomes price",
        figsize=(14.5, 9.0),
    )
    figure.savefig(outdir / "orderwave-built-in-overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_current_book(outdir: Path) -> None:
    market = Market(seed=42, capture="visual")
    market.run(steps=720)
    figure = market.plot_book(levels=10, title="Current order book snapshot", figsize=(11, 6.8))
    figure.savefig(outdir / "orderwave-built-in-current-book.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_diagnostics(outdir: Path) -> None:
    market = Market(seed=11, capture="visual", config={"market_rate": 3.4, "cancel_rate": 4.8, "fair_price_vol": 0.45})
    market.run(steps=1_000)
    figure = market.plot_heatmap(
        anchor="price",
        max_steps=900,
        price_window_ticks=12,
        title="Regime-aware level-ranked signed depth heatmap",
        figsize=(13.5, 7.5),
    )
    figure.savefig(outdir / "orderwave-built-in-diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_variants(outdir: Path) -> None:
    variants = {
        "default": {},
        "slower tape": {"market_rate": 1.2, "fair_price_vol": 0.20},
        "faster tape": {"market_rate": 3.8, "cancel_rate": 5.0, "fair_price_vol": 0.45},
    }

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for axis, (label, config) in zip(axes, variants.items(), strict=True):
        market = Market(seed=42, config=config)
        history = market.run(steps=320).history
        axis.plot(history["step"], history["mid_price"], color="#0f172a", linewidth=1.6, label="mid")
        axis.plot(history["step"], history["fair_price"], color="#0f766e", linewidth=1.0, alpha=0.85, label="fair")
        axis.fill_between(
            history["step"],
            history["best_bid"],
            history["best_ask"],
            color="#94a3b8",
            alpha=0.22,
        )
        axis.set_title(label)
        axis.set_xlabel("Step")
        axis.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="upper left", frameon=False)
    figure.suptitle("Configuration variants")
    figure.savefig(outdir / "orderwave-built-in-presets.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    render_overview(args.outdir)
    render_current_book(args.outdir)
    render_diagnostics(args.outdir)
    render_variants(args.outdir)


if __name__ == "__main__":
    main()
