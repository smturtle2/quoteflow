from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

from orderwave import Market

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
    heatmap = market.plot_heatmap(anchor="price", max_steps=60, price_window_ticks=12)
    book = market.plot_book(levels=8)

    assert isinstance(overview, Figure)
    assert isinstance(heatmap, Figure)
    assert isinstance(book, Figure)
