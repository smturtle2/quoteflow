from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from orderwave import Market
from orderwave.visualization import _prepare_heatmap

matplotlib.use("Agg")


def test_visual_capture_is_reproducible() -> None:
    market_a = Market(seed=21, capture="visual")
    market_b = Market(seed=21, capture="visual")

    market_a.run(steps=120)
    market_b.run(steps=120)

    np.testing.assert_allclose(
        market_a._visual_history.signed_depth_matrix(),  # type: ignore[union-attr]
        market_b._visual_history.signed_depth_matrix(),  # type: ignore[union-attr]
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        market_a._visual_history.center_ticks_array(),  # type: ignore[union-attr]
        market_b._visual_history.center_ticks_array(),  # type: ignore[union-attr]
    )


def test_plot_methods_require_visual_capture_for_heatmap() -> None:
    market = Market(seed=42)
    market.run(steps=30)

    with pytest.raises(RuntimeError, match="capture='visual'"):
        market.plot()
    with pytest.raises(RuntimeError, match="capture='visual'"):
        market.plot_heatmap()

    assert isinstance(market.plot_book(), Figure)


def test_plot_methods_return_figures_in_visual_mode() -> None:
    market = Market(seed=42, capture="visual")
    market.run(steps=80)

    overview = market.plot(max_steps=60, price_window_ticks=10)
    heatmap = market.plot_heatmap(anchor="price", max_steps=60, price_window_ticks=5)
    book = market.plot_book(levels=8)
    axis = heatmap.axes[0]
    labels = [tick.get_text() for tick in axis.get_yticklabels()]

    assert isinstance(overview, Figure)
    assert isinstance(heatmap, Figure)
    assert isinstance(book, Figure)
    assert axis.get_ylabel() == "Levels"
    assert labels == ["ask 5", "ask 4", "ask 3", "ask 2", "ask 1", "bid 1", "bid 2", "bid 3", "bid 4", "bid 5"]
    assert len(axis.lines) == 1


def test_heatmap_rows_are_stable_level_ranks() -> None:
    market = Market(seed=42, capture="visual")
    market.run(steps=120)
    store = market._visual_history

    assert store is not None

    mid_payload = _prepare_heatmap(
        store=store,
        tick_size=market.tick_size,
        anchor="mid",
        max_steps=60,
        price_window_ticks=5,
    )
    price_payload = _prepare_heatmap(
        store=store,
        tick_size=market.tick_size,
        anchor="price",
        max_steps=60,
        price_window_ticks=5,
    )

    np.testing.assert_allclose(mid_payload.signed_depth, price_payload.signed_depth, equal_nan=True)
    assert mid_payload.yticklabels == ["ask 5", "ask 4", "ask 3", "ask 2", "ask 1", "bid 1", "bid 2", "bid 3", "bid 4", "bid 5"]
    assert price_payload.yticklabels == mid_payload.yticklabels

    ask_values = price_payload.signed_depth[:5][np.isfinite(price_payload.signed_depth[:5])]
    bid_values = price_payload.signed_depth[5:][np.isfinite(price_payload.signed_depth[5:])]
    assert ask_values.size > 0
    assert bid_values.size > 0
    assert np.all(ask_values < 0.0)
    assert np.all(bid_values > 0.0)
