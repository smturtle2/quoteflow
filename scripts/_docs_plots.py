from __future__ import annotations

"""Documentation-only plotting helpers."""

from typing import Mapping

import numpy as np
import pandas as pd


def plot_preset_comparison(
    histories: Mapping[str, pd.DataFrame],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
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

    colors = {
        "balanced": "#0f766e",
        "trend": "#f97316",
        "volatile": "#dc2626",
    }
    ordered = [name for name in ("balanced", "trend", "volatile") if name in histories]
    if not ordered:
        raise ValueError("histories must include at least one preset")

    figure, axes = plt.subplots(1, len(ordered), figsize=figsize or (16, 4.8), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    figure.suptitle(title or "Preset behaviors at a glance", fontsize=16, fontweight="bold")
    legend_handles = None
    legend_labels = None

    for axis, preset in zip(axes_array, ordered):
        history = histories[preset]
        steps = history["step"].to_numpy()
        mid = history["mid_price"].to_numpy()
        last = history["last_price"].to_numpy()
        spread = history["spread"].to_numpy()
        mid_ret = history["mid_price"].diff().fillna(0.0)
        color = colors[preset]

        axis.plot(steps, mid, color=color, linewidth=1.8, label="Mid")
        axis.plot(steps, last, color="#0f172a", linewidth=0.95, alpha=0.75, label="Last")
        axis.fill_between(steps, mid - (spread / 2.0), mid + (spread / 2.0), color=color, alpha=0.12)
        axis.set_title(preset.capitalize())
        axis.set_xlabel("Step")
        axis.grid(alpha=0.25, linestyle="--")
        if axis is axes_array[0]:
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
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=2,
            frameon=False,
        )

    return figure
