from __future__ import annotations

"""Plotting helpers and compact visual history storage."""

from array import array
from dataclasses import dataclass

import numpy as np
import pandas as pd

from orderwave.book import OrderBook
from orderwave.utils import tick_to_price


class VisualHistoryStore:
    """Fixed-window signed depth history for overview and heatmap plots."""

    def __init__(self, *, depth_window_ticks: int) -> None:
        if depth_window_ticks < 1:
            raise ValueError("depth_window_ticks must be positive")
        self.depth_window_ticks = int(depth_window_ticks)
        self._step = array("I")
        self._center_tick = array("i")
        self._best_bid_tick = array("i")
        self._best_ask_tick = array("i")
        self._signed_depth = array("f")

    def append(
        self,
        *,
        step: int,
        center_tick: int,
        best_bid_tick: int,
        best_ask_tick: int,
        signed_depth: np.ndarray,
    ) -> None:
        expected_width = (2 * self.depth_window_ticks) + 1
        if signed_depth.shape != (expected_width,):
            raise ValueError("signed_depth has unexpected width")
        self._step.append(int(step))
        self._center_tick.append(int(center_tick))
        self._best_bid_tick.append(int(best_bid_tick))
        self._best_ask_tick.append(int(best_ask_tick))
        self._signed_depth.extend(float(value) for value in signed_depth.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self._step)

    def steps_array(self) -> np.ndarray:
        return np.frombuffer(self._step, dtype=np.uint32).astype(np.int64, copy=False)

    def center_ticks_array(self) -> np.ndarray:
        return np.frombuffer(self._center_tick, dtype=np.int32).astype(np.int64, copy=False)

    def best_bid_ticks_array(self) -> np.ndarray:
        return np.frombuffer(self._best_bid_tick, dtype=np.int32).astype(np.int64, copy=False)

    def best_ask_ticks_array(self) -> np.ndarray:
        return np.frombuffer(self._best_ask_tick, dtype=np.int32).astype(np.int64, copy=False)

    def signed_depth_matrix(self) -> np.ndarray:
        width = (2 * self.depth_window_ticks) + 1
        if len(self) == 0:
            return np.empty((width, 0), dtype=np.float32)
        matrix = np.frombuffer(self._signed_depth, dtype=np.float32).reshape(len(self), width)
        return matrix.T


@dataclass(frozen=True)
class HeatmapPayload:
    steps: np.ndarray
    signed_depth: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    best_bid_trace: np.ndarray
    best_ask_trace: np.ndarray
    ylabel: str
    yticks: np.ndarray
    yticklabels: list[str]


def plot_order_book(
    book: OrderBook,
    *,
    tick_size: float,
    levels: int,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render the current order-book snapshot."""

    if levels <= 0:
        raise ValueError("levels must be positive")

    plt = _configure_matplotlib()
    bids = book.levels("bid", levels)
    asks = book.levels("ask", levels)
    if not bids or not asks:
        raise ValueError("order book plot requires both bid and ask liquidity")
    best_bid_tick = book.best_bid_tick
    best_ask_tick = book.best_ask_tick
    if best_bid_tick is None or best_ask_tick is None:
        raise ValueError("order book plot requires both bid and ask liquidity")

    bid_prices = np.array([tick_to_price(tick, tick_size) for tick, _ in bids], dtype=float)
    ask_prices = np.array([tick_to_price(tick, tick_size) for tick, _ in asks], dtype=float)
    bid_qty = np.array([qty for _, qty in bids], dtype=float)
    ask_qty = np.array([qty for _, qty in asks], dtype=float)
    max_qty = max(float(np.max(bid_qty)), float(np.max(ask_qty)), 1.0)

    figure, axis = plt.subplots(figsize=figsize or (11, 6.5), constrained_layout=True)
    axis.barh(bid_prices, -bid_qty, color="#2563eb", alpha=0.90, height=tick_size * 0.85, label="Bid depth")
    axis.barh(ask_prices, ask_qty, color="#dc2626", alpha=0.88, height=tick_size * 0.85, label="Ask depth")
    axis.axvline(0.0, color="#0f172a", linewidth=1.0, alpha=0.35)
    axis.axhline(tick_to_price(best_bid_tick, tick_size), color="#1d4ed8", linestyle="--", linewidth=1.0)
    axis.axhline(tick_to_price(best_ask_tick, tick_size), color="#b91c1c", linestyle="--", linewidth=1.0)
    axis.set_xlim(-(max_qty * 1.15), max_qty * 1.15)
    axis.set_xlabel("Aggregate depth")
    axis.set_ylabel("Price")
    axis.set_title(title or "Order book snapshot")
    axis.grid(alpha=0.25, linestyle="--")
    axis.legend(loc="upper left", frameon=False)
    axis.xaxis.set_major_formatter(_absolute_axis_formatter())
    return figure


def plot_heatmap(
    store: VisualHistoryStore,
    *,
    tick_size: float,
    anchor: str,
    max_steps: int,
    price_window_ticks: int | None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render a standalone signed-depth heatmap."""

    plt = _configure_matplotlib()
    figure, axis = plt.subplots(figsize=figsize or (13, 7.5), constrained_layout=True)
    _draw_heatmap(
        axis=axis,
        store=store,
        tick_size=tick_size,
        anchor=anchor,
        max_steps=max_steps,
        price_window_ticks=price_window_ticks,
    )
    axis.set_title(title or ("Price-anchored heatmap" if anchor == "price" else "Mid-anchored heatmap"))
    return figure


def plot_market_overview(
    history: pd.DataFrame,
    store: VisualHistoryStore,
    *,
    tick_size: float,
    max_steps: int,
    price_window_ticks: int | None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Render a two-panel market overview with price path and heatmap."""

    plt = _configure_matplotlib()
    figure, (price_axis, heat_axis) = plt.subplots(
        2,
        1,
        figsize=figsize or (14, 9),
        sharex=False,
        height_ratios=(1.0, 1.7),
        constrained_layout=True,
    )

    plot_history = history
    if max_steps > 0 and len(history) > max_steps:
        step_array = history["step"].to_numpy(dtype=int)
        groups = _downsample_groups(len(step_array), max_steps)
        rows = [group[-1] for group in groups if len(group) > 0]
        plot_history = history.iloc[rows].reset_index(drop=True)

    steps = plot_history["step"].to_numpy(dtype=float)
    price_axis.plot(steps, plot_history["mid_price"], color="#0f172a", linewidth=1.8, label="Mid")
    price_axis.plot(steps, plot_history["last_price"], color="#f97316", linewidth=1.1, alpha=0.9, label="Last")
    price_axis.plot(steps, plot_history["fair_price"], color="#0f766e", linewidth=1.2, alpha=0.85, label="Fair")
    price_axis.fill_between(
        steps,
        plot_history["best_bid"].to_numpy(dtype=float),
        plot_history["best_ask"].to_numpy(dtype=float),
        color="#94a3b8",
        alpha=0.22,
        label="Spread",
    )
    price_axis.set_ylabel("Price")
    price_axis.set_title(title or "orderwave overview")
    price_axis.grid(alpha=0.25, linestyle="--")
    price_axis.legend(loc="upper left", frameon=False, ncol=4)

    _draw_heatmap(
        axis=heat_axis,
        store=store,
        tick_size=tick_size,
        anchor="mid",
        max_steps=max_steps,
        price_window_ticks=price_window_ticks,
    )
    heat_axis.set_xlabel("Step")
    return figure


def _draw_heatmap(
    *,
    axis,
    store: VisualHistoryStore,
    tick_size: float,
    anchor: str,
    max_steps: int,
    price_window_ticks: int | None,
) -> None:
    data = _prepare_heatmap(
        store=store,
        tick_size=tick_size,
        anchor=anchor,
        max_steps=max_steps,
        price_window_ticks=price_window_ticks,
    )
    transformed, vmax = _scaled_signed_depth(data.signed_depth)
    if transformed.size == 0:
        raise ValueError("heatmap requires recorded visual history")

    mesh = axis.pcolormesh(
        data.x_edges,
        data.y_edges,
        np.ma.masked_invalid(transformed),
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
    )
    axis.plot(data.steps, data.best_bid_trace, color="#1d4ed8", linewidth=0.9, alpha=0.90)
    axis.plot(data.steps, data.best_ask_trace, color="#b91c1c", linewidth=0.9, alpha=0.90)
    axis.set_ylabel(data.ylabel)
    axis.grid(False)

    if data.yticks.size > 0:
        axis.set_yticks(data.yticks)
        axis.set_yticklabels(data.yticklabels)

    colorbar = axis.figure.colorbar(mesh, ax=axis, pad=0.015)
    colorbar.set_label("Signed depth (asinh-scaled)")


def _prepare_heatmap(
    *,
    store: VisualHistoryStore,
    tick_size: float,
    anchor: str,
    max_steps: int,
    price_window_ticks: int | None,
) -> HeatmapPayload:
    steps = store.steps_array()
    centers = store.center_ticks_array()
    best_bid = store.best_bid_ticks_array()
    best_ask = store.best_ask_ticks_array()
    signed_depth = store.signed_depth_matrix()

    if signed_depth.shape[1] == 0:
        raise ValueError("visual capture is empty")

    if max_steps > 0 and signed_depth.shape[1] > max_steps:
        groups = _downsample_groups(signed_depth.shape[1], max_steps)
        reduced_steps: np.ndarray = np.empty(len(groups), dtype=float)
        reduced_centers: np.ndarray = np.empty(len(groups), dtype=float)
        reduced_best_bid: np.ndarray = np.empty(len(groups), dtype=float)
        reduced_best_ask: np.ndarray = np.empty(len(groups), dtype=float)
        reduced_matrix: np.ndarray = np.empty((signed_depth.shape[0], len(groups)), dtype=np.float32)
        for column_index, group in enumerate(groups):
            reduced_steps[column_index] = float(steps[group[-1]])
            reduced_centers[column_index] = float(np.median(centers[group]))
            reduced_best_bid[column_index] = float(np.median(best_bid[group]))
            reduced_best_ask[column_index] = float(np.median(best_ask[group]))
            reduced_matrix[:, column_index] = _aggregate_signed_block(signed_depth[:, group])
        steps = reduced_steps.astype(float, copy=False)
        centers = reduced_centers.astype(float, copy=False)
        best_bid = reduced_best_bid.astype(float, copy=False)
        best_ask = reduced_best_ask.astype(float, copy=False)
        signed_depth = reduced_matrix
    else:
        steps = steps.astype(float, copy=False)
        centers = centers.astype(float, copy=False)
        best_bid = best_bid.astype(float, copy=False)
        best_ask = best_ask.astype(float, copy=False)

    max_window = store.depth_window_ticks
    window = max_window if price_window_ticks is None else max(1, min(int(price_window_ticks), max_window))
    offset_center = max_window
    row_slice = slice(offset_center - window, offset_center + window + 1)
    relative_matrix = signed_depth[row_slice]
    relative_offsets: np.ndarray = np.arange(-window, window + 1, dtype=float)

    if anchor == "mid":
        y_edges = np.arange(-window - 0.5, window + 1.5, 1.0)
        best_bid_trace = best_bid - centers
        best_ask_trace = best_ask - centers
        ytick_positions, ytick_labels = _sparse_tick_labels(relative_offsets, unit="")
        return HeatmapPayload(
            steps=steps,
            signed_depth=relative_matrix,
            x_edges=_step_edges(steps),
            y_edges=y_edges,
            best_bid_trace=best_bid_trace,
            best_ask_trace=best_ask_trace,
            ylabel="Relative ticks",
            yticks=ytick_positions,
            yticklabels=ytick_labels,
        )

    if price_window_ticks is None:
        absolute_ticks: np.ndarray = np.arange(
            int(np.floor(np.min(centers))) - window,
            int(np.ceil(np.max(centers))) + window + 1,
            dtype=int,
        )
    else:
        anchor_center = int(round(np.median(centers)))
        absolute_ticks = np.arange(anchor_center - window, anchor_center + window + 1, dtype=int)
    absolute_matrix: np.ndarray = np.full((absolute_ticks.size, steps.size), np.nan, dtype=np.float32)
    for column_index in range(steps.size):
        tick_base = int(round(float(centers[column_index]))) - window
        column = relative_matrix[:, column_index]
        for row_index, value in enumerate(column):
            if np.isnan(value):
                continue
            absolute_tick = tick_base + row_index
            target: int = absolute_tick - int(absolute_ticks[0])
            if 0 <= target < absolute_matrix.shape[0]:
                absolute_matrix[target, column_index] = value

    y_prices = np.array([tick_to_price(tick, tick_size) for tick in absolute_ticks], dtype=float)
    y_step = tick_size
    y_edges = np.concatenate(([y_prices[0] - (0.5 * y_step)], y_prices + (0.5 * y_step)))
    best_bid_trace = np.array([tick_to_price(tick, tick_size) for tick in best_bid], dtype=float)
    best_ask_trace = np.array([tick_to_price(tick, tick_size) for tick in best_ask], dtype=float)
    ytick_positions, ytick_labels = _sparse_tick_labels(y_prices, unit="price")
    return HeatmapPayload(
        steps=steps,
        signed_depth=absolute_matrix,
        x_edges=_step_edges(steps),
        y_edges=y_edges,
        best_bid_trace=best_bid_trace,
        best_ask_trace=best_ask_trace,
        ylabel="Price",
        yticks=ytick_positions,
        yticklabels=ytick_labels,
    )


def _aggregate_signed_block(block: np.ndarray) -> np.ndarray:
    result = np.full(block.shape[0], np.nan, dtype=np.float32)
    for row_index in range(block.shape[0]):
        row = block[row_index]
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            continue
        finite_values = row[finite_mask]
        result[row_index] = finite_values[int(np.argmax(np.abs(finite_values)))]
    return result


def _downsample_groups(length: int, target: int) -> list[np.ndarray]:
    if target <= 0 or length <= target:
        return [np.arange(length, dtype=int)]
    edges = np.linspace(0, length, num=target + 1, dtype=int)
    groups = [np.arange(edges[index], edges[index + 1], dtype=int) for index in range(target)]
    return [group for group in groups if group.size > 0]


def _step_edges(steps: np.ndarray) -> np.ndarray:
    if steps.size == 1:
        return np.array([steps[0] - 0.5, steps[0] + 0.5], dtype=float)
    deltas = np.diff(steps)
    left_edge = steps[0] - (deltas[0] / 2.0)
    right_edge = steps[-1] + (deltas[-1] / 2.0)
    mid_edges = steps[:-1] + (deltas / 2.0)
    return np.concatenate(([left_edge], mid_edges, [right_edge])).astype(float, copy=False)


def _scaled_signed_depth(signed_depth: np.ndarray) -> tuple[np.ndarray, float]:
    finite = signed_depth[np.isfinite(signed_depth)]
    if finite.size == 0:
        return signed_depth.astype(float, copy=False), 1.0
    scale = float(np.quantile(np.abs(finite), 0.90))
    if scale <= 0.0:
        scale = 1.0
    transformed = np.arcsinh(signed_depth / scale)
    transformed_finite = np.abs(transformed[np.isfinite(transformed)])
    vmax = float(np.quantile(transformed_finite, 0.995)) if transformed_finite.size else 1.0
    return transformed, max(vmax, 1e-6)


def _sparse_tick_labels(values: np.ndarray, *, unit: str) -> tuple[np.ndarray, list[str]]:
    if values.size <= 9:
        return values, [f"{value:.2f}" if unit == "price" else f"{int(value)}" for value in values]
    positions = np.linspace(0, values.size - 1, num=9, dtype=int)
    unique_positions = np.unique(positions)
    labels = []
    for position in unique_positions:
        value = values[position]
        labels.append(f"{value:.2f}" if unit == "price" else f"{int(value)}")
    return values[unique_positions], labels


def _configure_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _absolute_axis_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _: f"{abs(value):.0f}")


__all__ = ["VisualHistoryStore", "plot_heatmap", "plot_market_overview", "plot_order_book"]
