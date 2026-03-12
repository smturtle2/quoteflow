from __future__ import annotations

"""Compact history storage for summary snapshots."""

from typing import Any

import pandas as pd

SUMMARY_COLUMNS = [
    "step",
    "last_price",
    "mid_price",
    "best_bid",
    "best_ask",
    "spread",
    "bid_depth",
    "ask_depth",
    "depth_imbalance",
    "buy_aggr_volume",
    "sell_aggr_volume",
    "fair_price",
]


class HistoryBuffer:
    """Append-only store for summary snapshots."""

    def __init__(self) -> None:
        self._rows: list[tuple[Any, ...]] = []

    def append(self, snapshot: dict[str, object]) -> None:
        self._rows.append(tuple(snapshot[column] for column in SUMMARY_COLUMNS))

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._rows, columns=SUMMARY_COLUMNS)


__all__ = ["HistoryBuffer", "SUMMARY_COLUMNS"]
