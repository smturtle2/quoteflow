from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orderwave import Market
from orderwave.visualization import _plot_preset_comparison


def build_market(*, preset: str, seed: int, steps: int, levels: int) -> Market:
    market = Market(seed=seed, levels=levels, config={"preset": preset})
    market.gen(steps)
    return market


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
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    overview_market = build_market(preset="trend", seed=17, steps=720, levels=8)
    overview_figure = overview_market.plot(
        levels=8,
        title="Order flow becomes price",
        figsize=(15, 10.5),
    )
    overview_figure.savefig(args.outdir / "orderwave-built-in-overview.png", dpi=180, bbox_inches="tight")
    plt.close(overview_figure)

    current_book_figure = overview_market.plot_book(
        levels=8,
        title="Current order book snapshot",
        figsize=(11, 7),
    )
    current_book_figure.savefig(args.outdir / "orderwave-built-in-current-book.png", dpi=180, bbox_inches="tight")
    plt.close(current_book_figure)

    diagnostics_market = build_market(preset="balanced", seed=9, steps=4_000, levels=6)
    diagnostics_figure = diagnostics_market.plot_diagnostics(
        title="Microstructure diagnostics",
        figsize=(14, 8.5),
    )
    diagnostics_figure.savefig(args.outdir / "orderwave-built-in-diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(diagnostics_figure)

    preset_histories = {
        "balanced": build_market(preset="balanced", seed=42, steps=480, levels=6).get_history(),
        "trend": build_market(preset="trend", seed=42, steps=480, levels=6).get_history(),
        "volatile": build_market(preset="volatile", seed=42, steps=480, levels=6).get_history(),
    }
    preset_figure = _plot_preset_comparison(
        preset_histories,
        title="Preset behaviors at a glance",
        figsize=(16, 4.8),
    )
    preset_figure.savefig(args.outdir / "orderwave-built-in-presets.png", dpi=180, bbox_inches="tight")
    plt.close(preset_figure)

    print(f"rendered documentation images to {args.outdir}")


if __name__ == "__main__":
    main()
