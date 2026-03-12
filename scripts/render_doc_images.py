from __future__ import annotations

"""Regenerate documentation images for the compact orderwave API."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

from orderwave import Market


def render_overview(outdir: Path) -> None:
    market = Market(seed=42)
    history = market.run(steps=300).history

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(history["step"], history["mid_price"], label="mid")
    axis.plot(history["step"], history["last_price"], label="last", alpha=0.8)
    axis.plot(history["step"], history["fair_price"], label="fair", alpha=0.8)
    axis.set_title("Orderwave Price Path")
    axis.set_xlabel("Step")
    axis.set_ylabel("Price")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(outdir / "orderwave-built-in-overview.png", dpi=160)
    plt.close(figure)


def render_current_book(outdir: Path) -> None:
    market = Market(seed=42)
    market.run(steps=300)
    snapshot = market.get()

    figure, axis = plt.subplots(figsize=(9, 5))
    bid_prices = [level["price"] for level in snapshot["bids"]]
    bid_qtys = [level["qty"] for level in snapshot["bids"]]
    ask_prices = [level["price"] for level in snapshot["asks"]]
    ask_qtys = [level["qty"] for level in snapshot["asks"]]

    axis.barh(bid_prices, bid_qtys, color="#0f766e", alpha=0.85, label="bids")
    axis.barh(ask_prices, ask_qtys, color="#b91c1c", alpha=0.80, label="asks")
    axis.axhline(snapshot["mid_price"], color="#111827", linewidth=1.2, linestyle="--", label="mid")
    axis.set_title("Visible Aggregate Book")
    axis.set_xlabel("Quantity")
    axis.set_ylabel("Price")
    axis.legend()
    axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(outdir / "orderwave-built-in-current-book.png", dpi=160)
    plt.close(figure)


def render_diagnostics(outdir: Path) -> None:
    market = Market(seed=42)
    history = market.run(steps=300).history

    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(history["step"], history["spread"], color="#1d4ed8")
    axes[0].set_ylabel("Spread")
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["step"], history["depth_imbalance"], color="#7c3aed")
    axes[1].set_ylabel("Imbalance")
    axes[1].grid(alpha=0.25)

    axes[2].plot(history["step"], history["buy_aggr_volume"], label="buy aggr", color="#0f766e")
    axes[2].plot(history["step"], history["sell_aggr_volume"], label="sell aggr", color="#b91c1c")
    axes[2].set_ylabel("Aggressive Volume")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    figure.suptitle("Compact Diagnostics")
    figure.tight_layout()
    figure.savefig(outdir / "orderwave-built-in-diagnostics.png", dpi=160)
    plt.close(figure)


def render_variants(outdir: Path) -> None:
    variants = {
        "default": {},
        "slower tape": {"market_rate": 1.2, "fair_price_vol": 0.20},
        "faster tape": {"market_rate": 3.8, "cancel_rate": 5.0, "fair_price_vol": 0.45},
    }

    figure, axis = plt.subplots(figsize=(11, 5))
    for label, config in variants.items():
        market = Market(seed=42, config=config)
        history = market.run(steps=250).history
        axis.plot(history["step"], history["mid_price"], label=label)

    axis.set_title("Configuration Variants")
    axis.set_xlabel("Step")
    axis.set_ylabel("Mid Price")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(outdir / "orderwave-built-in-presets.png", dpi=160)
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
