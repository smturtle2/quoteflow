from __future__ import annotations

"""Regenerate documentation images for the current orderwave API."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

from orderwave import Market

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _corr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return 0.0
    if np.std(lhs) <= 1e-12 or np.std(rhs) <= 1e-12:
        return 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _sign_agreement(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size == 0 or rhs.size == 0:
        return 0.0
    lhs_sign = np.sign(lhs)
    rhs_sign = np.sign(rhs)
    mask = (lhs_sign != 0.0) & (rhs_sign != 0.0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(lhs_sign[mask] == rhs_sign[mask]))


def _doc_window_metrics(market: Market, steps: int) -> dict[str, float]:
    history = market.run(steps=steps).history
    mid_ticks = (history["mid_price"] / market.tick_size).to_numpy(dtype=float)
    returns = np.diff(mid_ticks, prepend=mid_ticks[0])
    signed_flow = (history["buy_aggr_volume"] - history["sell_aggr_volume"]).to_numpy(dtype=float)
    spread_ticks = (history["spread"] / market.tick_size).to_numpy(dtype=float)
    return {
        "abs_drift_ticks": float(abs(mid_ticks[-1] - mid_ticks[0])),
        "up_step_share": float(np.mean(returns > 0.0)),
        "down_step_share": float(np.mean(returns < 0.0)),
        "same_step_impact_corr": _corr(signed_flow[1:], returns[1:]),
        "flow_return_sign_agreement": _sign_agreement(signed_flow[1:], returns[1:]),
        "one_tick_spread_share": float(np.mean(spread_ticks == 1.0)),
        "wide_spread_share": float(np.mean(spread_ticks >= 4.0)),
    }


def _doc_seed_score(metrics: dict[str, float], *, spread_center: float = 0.62) -> float:
    return (
        metrics["abs_drift_ticks"]
        + (80.0 * abs(metrics["up_step_share"] - metrics["down_step_share"]))
        + (30.0 * abs(metrics["one_tick_spread_share"] - spread_center))
        + (140.0 * metrics["wide_spread_share"])
        - (24.0 * metrics["same_step_impact_corr"])
        - (18.0 * metrics["flow_return_sign_agreement"])
    )


def _select_doc_seed(
    *,
    steps: int,
    config: dict[str, float] | None = None,
    seeds: range = range(1, 25),
    max_abs_drift_ticks: float = 35.0,
    max_step_imbalance: float = 0.06,
    min_same_step_impact: float = 0.28,
    min_flow_agreement: float = 0.70,
    min_one_tick_share: float = 0.50,
    max_one_tick_share: float = 0.78,
    max_wide_share: float = 0.08,
) -> tuple[int, dict[str, float]]:
    accepted: list[tuple[float, int, dict[str, float]]] = []
    best_seed: int | None = None
    best_metrics: dict[str, float] | None = None
    best_score = float("inf")

    for seed in seeds:
        market = Market(seed=seed, config=config)
        try:
            metrics = _doc_window_metrics(market, steps)
        except RuntimeError:
            continue
        score = _doc_seed_score(metrics)
        if score < best_score:
            best_seed = seed
            best_metrics = metrics
            best_score = score
        if (
            metrics["abs_drift_ticks"] <= max_abs_drift_ticks
            and abs(metrics["up_step_share"] - metrics["down_step_share"]) <= max_step_imbalance
            and metrics["same_step_impact_corr"] >= min_same_step_impact
            and metrics["flow_return_sign_agreement"] >= min_flow_agreement
            and min_one_tick_share <= metrics["one_tick_spread_share"] <= max_one_tick_share
            and metrics["wide_spread_share"] <= max_wide_share
        ):
            accepted.append((score, seed, metrics))

    if not accepted:
        raise RuntimeError(
            "no documentation seed met the acceptance criteria; "
            f"best candidate was seed={best_seed} with metrics={best_metrics}"
        )
    accepted.sort(key=lambda item: item[0])
    _, seed, metrics = accepted[0]
    return seed, metrics


def render_overview(outdir: Path, *, seed: int) -> None:
    market = Market(seed=seed, capture="visual")
    market.run(steps=720)
    figure = market.plot(
        max_steps=720,
        price_window_ticks=12,
        title="Dynamic distribution synthesis becomes visible price and depth",
        figsize=(14.5, 9.0),
    )
    figure.savefig(outdir / "orderwave-built-in-overview.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_current_book(outdir: Path, *, seed: int) -> None:
    market = Market(seed=seed, capture="visual")
    market.run(steps=720)
    figure = market.plot_book(levels=10, title="Current order book snapshot", figsize=(11, 6.8))
    figure.savefig(outdir / "orderwave-built-in-current-book.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_diagnostics(outdir: Path) -> None:
    config = {"market_rate": 3.4, "cancel_rate": 4.8, "fair_price_vol": 0.45}
    seed, _ = _select_doc_seed(
        steps=1_000,
        config=config,
        seeds=range(1, 25),
        max_abs_drift_ticks=55.0,
        max_step_imbalance=0.08,
        min_same_step_impact=0.24,
        min_flow_agreement=0.68,
        max_one_tick_share=0.82,
        max_wide_share=0.14,
    )
    print(f"selected diagnostics seed={seed}")
    market = Market(seed=seed, capture="visual", config=config)
    market.run(steps=1_000)
    figure = market.plot_heatmap(
        anchor="mid",
        max_steps=900,
        price_window_ticks=12,
        title="Distribution-synthesized level-ranked signed depth heatmap",
        figsize=(13.5, 7.5),
    )
    figure.savefig(outdir / "orderwave-built-in-diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_variants(outdir: Path, *, seed: int) -> None:
    variants = {
        "default": {},
        "slower tape": {"market_rate": 1.2, "fair_price_vol": 0.20},
        "faster tape": {"market_rate": 3.8, "cancel_rate": 5.0, "fair_price_vol": 0.45},
    }

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, constrained_layout=True)
    for axis, (label, config) in zip(axes, variants.items(), strict=True):
        market = Market(seed=seed, config=config)
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
    figure.suptitle("Configuration variants under dynamic distribution synthesis")
    figure.savefig(outdir / "orderwave-built-in-presets.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    base_seed, metrics = _select_doc_seed(steps=720)
    print(f"selected overview seed={base_seed} metrics={metrics}")
    render_overview(args.outdir, seed=base_seed)
    render_current_book(args.outdir, seed=base_seed)
    render_diagnostics(args.outdir)
    render_variants(args.outdir, seed=base_seed)


if __name__ == "__main__":
    main()
