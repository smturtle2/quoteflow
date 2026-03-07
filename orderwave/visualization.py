from __future__ import annotations

"""Internal plotting helpers for built-in orderwave visualizations."""

from array import array
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from orderwave.book import OrderBook
from orderwave.utils import tick_to_price


REGIME_ORDER = ("calm", "directional", "stressed")


@dataclass(frozen=True)
class VisualHistoryRow:
    """Fixed-depth book state captured for plotting."""

    step: int
    bid_qty: np.ndarray
    ask_qty: np.ndarray


class VisualHistoryStore:
    """Columnar visible-book history used by market overview plots."""

    def __init__(self, *, depth: int) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.depth = int(depth)
        self._step = array("I")
        self._bid_qty = array("f")
        self._ask_qty = array("f")

    def append(
        self,
        *,
        step: int,
        bid_levels: Sequence[tuple[int, int]],
        ask_levels: Sequence[tuple[int, int]],
    ) -> None:
        self._step.append(int(step))
        self._extend_side(self._bid_qty, bid_levels)
        self._extend_side(self._ask_qty, ask_levels)

    def __len__(self) -> int:
        return len(self._step)

    def __iter__(self):
        bid_matrix = self.bid_matrix()
        ask_matrix = self.ask_matrix()
        steps = self.steps_array()
        for index in range(len(self)):
            yield VisualHistoryRow(
                step=int(steps[index]),
                bid_qty=bid_matrix[index].astype(float, copy=False),
                ask_qty=ask_matrix[index].astype(float, copy=False),
            )

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, stride = item.indices(len(self))
            if stride != 1:
                return [self[index] for index in range(start, stop, stride)]
            clone = VisualHistoryStore(depth=self.depth)
            if start >= stop:
                return clone
            bid_matrix = self.bid_matrix()[start:stop]
            ask_matrix = self.ask_matrix()[start:stop]
            clone._step.extend(int(value) for value in self.steps_array()[start:stop])
            clone._bid_qty.extend(float(value) for value in bid_matrix.reshape(-1))
            clone._ask_qty.extend(float(value) for value in ask_matrix.reshape(-1))
            return clone
        index = int(item)
        bid_matrix = self.bid_matrix()
        ask_matrix = self.ask_matrix()
        steps = self.steps_array()
        return VisualHistoryRow(
            step=int(steps[index]),
            bid_qty=bid_matrix[index].astype(float, copy=False),
            ask_qty=ask_matrix[index].astype(float, copy=False),
        )

    def steps_array(self) -> np.ndarray:
        return np.frombuffer(self._step, dtype=np.uint32).astype(np.int64, copy=False)

    def bid_matrix(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, self.depth), dtype=np.float32)
        return np.frombuffer(self._bid_qty, dtype=np.float32).reshape(len(self), self.depth)

    def ask_matrix(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, self.depth), dtype=np.float32)
        return np.frombuffer(self._ask_qty, dtype=np.float32).reshape(len(self), self.depth)

    def _extend_side(self, target: array, levels: Sequence[tuple[int, int]]) -> None:
        written = 0
        for _, qty in levels[: self.depth]:
            target.append(float(qty))
            written += 1
        for _ in range(written, self.depth):
            target.append(float("nan"))


@dataclass(frozen=True)
class DepthHeatmap:
    """Signed visible-depth matrix used by overview plots."""

    steps: np.ndarray
    ask_levels: int
    row_labels: list[str]
    signed_depth: np.ndarray


def capture_visual_history_row(book: OrderBook, *, step: int, depth: int) -> VisualHistoryRow:
    """Capture a fixed-width book slice for later plotting."""

    if depth <= 0:
        raise ValueError("depth must be positive")

    bid_qty = np.full(depth, np.nan, dtype=float)
    ask_qty = np.full(depth, np.nan, dtype=float)

    for index, (_, qty) in enumerate(book.top_levels("bid", depth)):
        bid_qty[index] = float(qty)
    for index, (_, qty) in enumerate(book.top_levels("ask", depth)):
        ask_qty[index] = float(qty)

    return VisualHistoryRow(step=int(step), bid_qty=bid_qty, ask_qty=ask_qty)


def build_depth_heatmap(rows: Sequence[VisualHistoryRow] | VisualHistoryStore, *, levels: int) -> DepthHeatmap:
    """Build a signed depth heatmap matrix from captured book rows."""

    if levels <= 0:
        raise ValueError("levels must be positive")
    if not rows:
        raise ValueError("rows must not be empty")

    if isinstance(rows, VisualHistoryStore):
        max_depth = rows.depth
        clipped_levels = min(levels, max_depth)
        steps = rows.steps_array()
        bid_matrix = rows.bid_matrix()[:, :clipped_levels]
        ask_matrix = rows.ask_matrix()[:, :clipped_levels]
        row_count = len(rows)
    else:
        max_depth = len(rows[0].bid_qty)
        clipped_levels = min(levels, max_depth)
        steps = np.array([row.step for row in rows], dtype=int)
        bid_matrix = np.vstack([row.bid_qty[:clipped_levels] for row in rows]).astype(float, copy=False)
        ask_matrix = np.vstack([row.ask_qty[:clipped_levels] for row in rows]).astype(float, copy=False)
        row_count = len(rows)
    ask_labels = [f"ask {level}" for level in range(clipped_levels, 0, -1)]
    bid_labels = [f"bid {level}" for level in range(1, clipped_levels + 1)]
    row_labels = ask_labels + bid_labels
    signed_depth = np.full((len(row_labels), row_count), np.nan, dtype=float)

    for depth_index in range(clipped_levels):
        ask_values = ask_matrix[:, depth_index].astype(float, copy=False)
        bid_values = bid_matrix[:, depth_index].astype(float, copy=False)
        ask_target = clipped_levels - depth_index - 1
        bid_target = clipped_levels + depth_index
        signed_depth[ask_target, np.isfinite(ask_values)] = -ask_values[np.isfinite(ask_values)]
        signed_depth[bid_target, np.isfinite(bid_values)] = bid_values[np.isfinite(bid_values)]

    return DepthHeatmap(
        steps=steps,
        ask_levels=clipped_levels,
        row_labels=row_labels,
        signed_depth=signed_depth,
    )


def plot_market_overview(
    history: pd.DataFrame,
    rows: Sequence[VisualHistoryRow],
    *,
    levels: int,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render the built-in overview plot."""

    plt = _configure_matplotlib()
    row_count = min(len(history), len(rows))
    if row_count == 0:
        raise ValueError("overview plot requires recorded history")

    history_view = history.iloc[:row_count]
    heatmap = build_depth_heatmap(rows[:row_count], levels=levels)
    steps = history_view["step"].to_numpy()
    mid_price = history_view["mid_price"].to_numpy()
    last_price = history_view["last_price"].to_numpy()
    best_bid = history_view["best_bid"].to_numpy()
    best_ask = history_view["best_ask"].to_numpy()
    trade_strength = history_view["trade_strength"].to_numpy()

    figure, (price_ax, strength_ax, heat_ax) = plt.subplots(
        3,
        1,
        figsize=figsize or (14, 10.5),
        sharex=True,
        height_ratios=(1.0, 0.75, 2.2),
        constrained_layout=True,
    )

    price_ax.plot(steps, mid_price, color="#0f172a", linewidth=1.8, label="Mid price")
    price_ax.plot(steps, last_price, color="#f97316", linewidth=1.15, alpha=0.95, label="Last trade")
    price_ax.fill_between(steps, best_bid, best_ask, color="#94a3b8", alpha=0.2, label="Spread")
    price_ax.set_ylabel("Price")
    price_ax.set_title(title or "orderwave market overview")
    price_ax.grid(alpha=0.3, linestyle="--")
    price_ax.legend(loc="upper left", frameon=False, ncol=3)

    strength_ax.axhline(0.0, color="#0f172a", linewidth=0.85, alpha=0.45)
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
    strength_ax.plot(steps, trade_strength, color="#0f172a", linewidth=0.95, label="Trade strength")
    strength_ax.set_ylabel("Strength")
    strength_ax.set_ylim(-1.05, 1.05)
    strength_ax.grid(alpha=0.25, linestyle="--")
    strength_ax.legend(loc="upper left", frameon=False)

    cmap, norm, max_abs_depth = _signed_depth_style(heatmap.signed_depth)
    x_edges = np.arange(len(heatmap.steps) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(heatmap.row_labels) + 1, dtype=float) - 0.5
    mesh = heat_ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(heatmap.signed_depth),
        cmap=cmap,
        norm=norm,
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
    colorbar.set_label("Signed visible depth")
    colorbar.set_ticks(_colorbar_ticks(max_abs_depth))

    return figure


def plot_order_book(
    book: OrderBook,
    *,
    tick_size: float,
    levels: int,
    microprice: float,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render the current order-book snapshot with a real price axis."""

    if levels <= 0:
        raise ValueError("levels must be positive")

    plt = _configure_matplotlib()
    bids = book.top_levels("bid", levels)
    asks = book.top_levels("ask", levels)
    if not bids or not asks:
        raise ValueError("order book plot requires both bid and ask liquidity")

    bid_prices = np.array([tick_to_price(tick, tick_size) for tick, _ in bids], dtype=float)
    ask_prices = np.array([tick_to_price(tick, tick_size) for tick, _ in asks], dtype=float)
    bid_qty = np.array([qty for _, qty in bids], dtype=float)
    ask_qty = np.array([qty for _, qty in asks], dtype=float)
    max_qty = max(float(np.max(bid_qty)), float(np.max(ask_qty)), 1.0)

    figure, axis = plt.subplots(figsize=figsize or (11, 7), constrained_layout=True)
    axis.barh(bid_prices, -bid_qty, color="#38bdf8", alpha=0.9, height=tick_size * 0.8, label="Bid depth")
    axis.barh(ask_prices, ask_qty, color="#ef4444", alpha=0.9, height=tick_size * 0.8, label="Ask depth")
    axis.axvline(0.0, color="#0f172a", linewidth=1.0, alpha=0.5)

    best_bid = tick_to_price(book.best_bid_tick, tick_size)
    best_ask = tick_to_price(book.best_ask_tick, tick_size)
    axis.axhline(best_bid, color="#0284c7", linestyle="--", linewidth=1.0, alpha=0.9, label="Best bid")
    axis.axhline(best_ask, color="#dc2626", linestyle="--", linewidth=1.0, alpha=0.9, label="Best ask")
    axis.axhline(microprice, color="#f97316", linestyle=":", linewidth=1.2, alpha=0.95, label="Microprice")

    axis.set_xlim(-(max_qty * 1.15), max_qty * 1.15)
    axis.set_xlabel("Aggregate depth")
    axis.set_ylabel("Price")
    axis.set_title(title or "orderwave order book snapshot")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(loc="upper left", frameon=False, ncol=2)
    axis.xaxis.set_major_formatter(_absolute_axis_formatter())

    return figure


def plot_market_diagnostics(
    history: pd.DataFrame,
    event_history: pd.DataFrame,
    debug_history: pd.DataFrame,
    *,
    imbalance_bins: int = 8,
    max_lag: int = 12,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render quality diagnostics for a simulated path."""

    if len(history) < 2:
        raise ValueError("diagnostics plot requires at least two history rows")
    if imbalance_bins < 2:
        raise ValueError("imbalance_bins must be at least 2")
    if max_lag < 1:
        raise ValueError("max_lag must be positive")

    plt = _configure_matplotlib()
    mid_ret = history["mid_price"].diff().fillna(0.0)
    next_ret = mid_ret.shift(-1).fillna(0.0)
    spread = history["spread"]
    imbalance = history["depth_imbalance"].clip(-1.0, 1.0)

    phase_order = ["open", "mid", "close"]
    phase_spread = history.groupby("session_phase")["spread"].mean().reindex(phase_order, fill_value=0.0)
    if not event_history.empty:
        phase_activity = (
            event_history.groupby(["session_phase", "step"])["fill_qty"]
            .sum()
            .groupby("session_phase")
            .mean()
            .reindex(phase_order, fill_value=0.0)
        )
    else:
        phase_activity = pd.Series(0.0, index=phase_order)

    bins = np.linspace(-1.0, 1.0, imbalance_bins + 1)
    bin_index = np.digitize(imbalance, bins[1:-1], right=False)
    binned_signal = pd.DataFrame({"bin": bin_index, "next_ret": next_ret}).groupby("bin")["next_ret"].mean()
    x_positions = np.arange(len(bins) - 1)
    y_values = np.array([float(binned_signal.get(index, 0.0)) for index in x_positions], dtype=float)
    x_labels = [f"{bins[index]:.2f}\n{bins[index + 1]:.2f}" for index in x_positions]

    market_events = event_history.loc[event_history["event_type"] == "market"].copy()
    buy_count_acf = np.zeros(max_lag, dtype=float)
    sell_count_acf = np.zeros(max_lag, dtype=float)
    if not market_events.empty:
        buy_counts = (
            market_events.assign(is_buy=(market_events["side"] == "buy").astype(float))
            .groupby("step")["is_buy"]
            .sum()
        )
        sell_counts = (
            market_events.assign(is_sell=(market_events["side"] == "sell").astype(float))
            .groupby("step")["is_sell"]
            .sum()
        )
        buy_count_acf = np.array([_safe_autocorr(buy_counts, lag) for lag in range(1, max_lag + 1)], dtype=float)
        sell_count_acf = np.array([_safe_autocorr(sell_counts, lag) for lag in range(1, max_lag + 1)], dtype=float)

    vol = history["realized_vol"].fillna(0.0)
    if vol.nunique() > 1:
        vol_bins = pd.qcut(vol.rank(method="first"), q=min(5, len(vol)), duplicates="drop")
        spread_vol = history.groupby(vol_bins, observed=False)["spread"].mean()
        spread_vol_x = np.arange(len(spread_vol))
        spread_vol_labels = [f"q{index + 1}" for index in range(len(spread_vol))]
        spread_vol_y = spread_vol.to_numpy(dtype=float)
    else:
        spread_vol_x = np.array([0], dtype=float)
        spread_vol_labels = ["flat"]
        spread_vol_y = np.array([float(spread.mean())], dtype=float)

    resiliency_curve = _depletion_resiliency_curve(history)
    shock_share = (
        debug_history["shock_state"].value_counts(normalize=True)
        if not debug_history.empty
        else pd.Series(dtype=float)
    )
    regime_share = history["regime"].value_counts(normalize=True).reindex(REGIME_ORDER, fill_value=0.0)

    figure, axes = plt.subplots(3, 2, figsize=figsize or (15, 10.5), constrained_layout=True)
    figure.suptitle(title or "orderwave diagnostics", fontsize=16, fontweight="bold")

    phase_x = np.arange(len(phase_order))
    axes[0, 0].bar(phase_x, phase_spread.to_numpy(dtype=float), color="#0f766e", width=0.6, label="Mean spread")
    phase_ax2 = axes[0, 0].twinx()
    phase_ax2.plot(
        phase_x,
        phase_activity.to_numpy(dtype=float),
        color="#f97316",
        linewidth=1.6,
        marker="o",
        label="Filled volume / step",
    )
    axes[0, 0].set_title("Session phase profile")
    axes[0, 0].set_xticks(phase_x)
    axes[0, 0].set_xticklabels([label.capitalize() for label in phase_order])
    axes[0, 0].set_ylabel("Spread")
    phase_ax2.set_ylabel("Filled volume / step")
    axes[0, 0].grid(alpha=0.25, linestyle="--")

    axes[0, 1].plot(x_positions, y_values, color="#2563eb", linewidth=1.8, marker="o")
    axes[0, 1].axhline(0.0, color="#0f172a", linewidth=0.85, alpha=0.45)
    axes[0, 1].set_title("Depth imbalance -> next mid return")
    axes[0, 1].set_xlabel("Imbalance bin")
    axes[0, 1].set_ylabel("Mean next return")
    axes[0, 1].set_xticks(x_positions)
    axes[0, 1].set_xticklabels(x_labels)
    axes[0, 1].grid(alpha=0.25, linestyle="--")

    lags = np.arange(1, max_lag + 1)
    axes[1, 0].plot(lags, buy_count_acf, color="#2563eb", linewidth=1.8, marker="o", label="Buy count")
    axes[1, 0].plot(lags, sell_count_acf, color="#dc2626", linewidth=1.6, marker="o", label="Sell count")
    axes[1, 0].axhline(0.0, color="#0f172a", linewidth=0.85, alpha=0.45)
    axes[1, 0].set_title("Market-flow excitation")
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].set_ylabel("Autocorr")
    axes[1, 0].grid(alpha=0.25, linestyle="--")
    axes[1, 0].legend(loc="upper right", frameon=False)

    axes[1, 1].plot(spread_vol_x, spread_vol_y, color="#dc2626", linewidth=1.8, marker="o")
    axes[1, 1].set_title("Spread-volatility coupling")
    axes[1, 1].set_xlabel("Volatility bin")
    axes[1, 1].set_ylabel("Mean spread")
    axes[1, 1].set_xticks(spread_vol_x)
    axes[1, 1].set_xticklabels(spread_vol_labels)
    axes[1, 1].grid(alpha=0.25, linestyle="--")

    axes[2, 0].plot(np.arange(len(resiliency_curve)), resiliency_curve, color="#0891b2", linewidth=1.9, marker="o")
    axes[2, 0].axhline(1.0, color="#0f172a", linewidth=0.85, alpha=0.35)
    axes[2, 0].set_title("Depletion resiliency")
    axes[2, 0].set_xlabel("Lag after depletion")
    axes[2, 0].set_ylabel("Depth recovery ratio")
    axes[2, 0].grid(alpha=0.25, linestyle="--")

    occupancy_names = list(REGIME_ORDER) + [name for name in shock_share.index if name != "none"]
    occupancy_values = list(regime_share.values) + [float(shock_share[name]) for name in shock_share.index if name != "none"]
    occupancy_colors = ["#0f766e", "#f97316", "#dc2626"] + ["#475569"] * max(0, len(occupancy_values) - 3)
    axes[2, 1].bar(occupancy_names, occupancy_values, color=occupancy_colors[: len(occupancy_values)], width=0.65)
    axes[2, 1].set_title("Regime and shock occupancy")
    axes[2, 1].set_ylabel("Share")
    axes[2, 1].set_ylim(0.0, max(1.0, max(occupancy_values, default=0.0) * 1.15))
    axes[2, 1].grid(axis="y", alpha=0.25, linestyle="--")

    return figure


def _plot_preset_comparison(
    histories: Mapping[str, pd.DataFrame],
    *,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render a docs-only preset comparison figure."""

    plt = _configure_matplotlib()
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
        figure.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=2, frameon=False)

    return figure


def _configure_matplotlib():
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
    return plt


def _signed_depth_style(values: np.ndarray):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    finite = values[np.isfinite(values)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 1.0
    if max_abs <= 0.0:
        max_abs = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "orderwave_signed_depth",
        ["#ef4444", "#e5e7eb", "#38bdf8"],
        N=256,
    )
    cmap.set_bad("#ffffff")
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    return cmap, norm, max_abs


def _colorbar_ticks(max_abs: float) -> list[float]:
    if max_abs <= 1.0:
        return [-1.0, 0.0, 1.0]
    return [round(value, 2) for value in np.linspace(-max_abs, max_abs, 5)]


def _absolute_axis_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _: f"{abs(value):.0f}")


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    if lag <= 0 or len(series) <= lag:
        return 0.0

    left = series.iloc[:-lag].to_numpy(dtype=float)
    right = series.iloc[lag:].to_numpy(dtype=float)
    if left.size < 2 or right.size < 2:
        return 0.0
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    if np.allclose(left_centered, 0.0) or np.allclose(right_centered, 0.0):
        return 0.0
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= 0.0:
        return 0.0

    return float(np.dot(left_centered, right_centered) / denom)


def _depletion_resiliency_curve(history: pd.DataFrame, *, horizon: int = 6) -> np.ndarray:
    if history.empty:
        return np.ones(horizon + 1, dtype=float)

    total_depth = (
        history["top_n_bid_qty"].to_numpy(dtype=float)
        + history["top_n_ask_qty"].to_numpy(dtype=float)
    )
    if len(total_depth) <= horizon + 1:
        return np.ones(horizon + 1, dtype=float)

    candidate_indices: list[int] = []
    for index in range(1, len(total_depth) - horizon):
        prev_depth = total_depth[index - 1]
        current_depth = total_depth[index]
        if prev_depth <= 0.0:
            continue
        drop_ratio = (prev_depth - current_depth) / prev_depth
        if drop_ratio >= 0.18:
            candidate_indices.append(index)

    if not candidate_indices:
        return np.ones(horizon + 1, dtype=float)

    curve = np.zeros(horizon + 1, dtype=float)
    for lag in range(horizon + 1):
        ratios = []
        for index in candidate_indices:
            baseline = total_depth[index - 1]
            if baseline <= 0.0:
                continue
            ratios.append(total_depth[index + lag] / baseline)
        curve[lag] = float(np.mean(ratios)) if ratios else 1.0

    return curve
