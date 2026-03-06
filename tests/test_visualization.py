from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from orderwave import Market
from orderwave.visualization import (
    VisualHistoryRow,
    _signed_depth_style,
    build_depth_heatmap,
)

matplotlib.use("Agg")


def test_plot_methods_return_figures_and_do_not_mutate_state() -> None:
    market = Market(seed=42, config={"preset": "trend"})
    market.gen(steps=30)
    snapshot_before = market.get()
    history_before = market.get_history()
    event_history_before = market.get_event_history()
    debug_history_before = market.get_debug_history()

    overview = market.plot()
    book = market.plot_book()
    diagnostics = market.plot_diagnostics()

    assert isinstance(overview, Figure)
    assert isinstance(book, Figure)
    assert isinstance(diagnostics, Figure)
    assert market.get() == snapshot_before
    pd.testing.assert_frame_equal(market.get_history(), history_before)
    pd.testing.assert_frame_equal(market.get_event_history(), event_history_before)
    pd.testing.assert_frame_equal(market.get_debug_history(), debug_history_before)


def test_plot_works_at_step_zero_and_book_plot_uses_real_price_axis() -> None:
    market = Market(seed=3)

    overview = market.plot()
    book = market.plot_book()

    assert isinstance(overview, Figure)
    assert isinstance(book, Figure)
    assert book.axes[0].get_ylabel() == "Price"


def test_plot_diagnostics_requires_at_least_two_rows() -> None:
    market = Market(seed=4)

    with pytest.raises(ValueError, match="at least two history rows"):
        market.plot_diagnostics()


def test_depth_heatmap_order_and_missing_levels_are_preserved() -> None:
    rows = [
        VisualHistoryRow(
            step=0,
            ask_qty=np.array([5.0, 3.0, np.nan]),
            bid_qty=np.array([7.0, np.nan, np.nan]),
        ),
        VisualHistoryRow(
            step=1,
            ask_qty=np.array([4.0, np.nan, np.nan]),
            bid_qty=np.array([6.0, 2.0, 1.0]),
        ),
    ]

    heatmap = build_depth_heatmap(rows, levels=3)

    assert heatmap.row_labels == ["ask 3", "ask 2", "ask 1", "bid 1", "bid 2", "bid 3"]
    assert np.isnan(heatmap.signed_depth[0, 0])
    assert heatmap.signed_depth[1, 0] == -3.0
    assert heatmap.signed_depth[2, 0] == -5.0
    assert heatmap.signed_depth[3, 1] == 6.0
    assert np.isnan(heatmap.signed_depth[4, 0])


def test_signed_depth_midpoint_is_light_gray_and_missing_cells_are_white() -> None:
    values = np.array([[-4.0, np.nan, 0.0, 3.0]], dtype=float)
    cmap, norm, _ = _signed_depth_style(values)

    midpoint = cmap(norm(0.0))
    bad = cmap(np.ma.masked_invalid([np.nan]))[0]

    assert midpoint[:3] != pytest.approx((0.0, 0.0, 0.0))
    assert midpoint[0] == pytest.approx(midpoint[1], abs=0.06)
    assert midpoint[1] == pytest.approx(midpoint[2], abs=0.06)
    assert bad[:3] == pytest.approx((1.0, 1.0, 1.0))


def test_cli_examples_render_with_agg_backend(tmp_path: Path) -> None:
    env = {**os.environ, "MPLBACKEND": "Agg"}
    doc_out = tmp_path / "docs-assets"
    plot_out = tmp_path / "overview.png"

    render_docs = subprocess.run(
        [sys.executable, "scripts/render_doc_images.py", "--outdir", str(doc_out)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    render_example = subprocess.run(
        [sys.executable, "examples/plot_market_heatmap.py", "--steps", "40", "--output", str(plot_out)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert render_docs.returncode == 0, render_docs.stderr
    assert render_example.returncode == 0, render_example.stderr
    assert (doc_out / "orderwave-built-in-overview.png").exists()
    assert (doc_out / "orderwave-built-in-current-book.png").exists()
    assert (doc_out / "orderwave-built-in-diagnostics.png").exists()
    assert (doc_out / "orderwave-built-in-presets.png").exists()
    assert plot_out.exists()
