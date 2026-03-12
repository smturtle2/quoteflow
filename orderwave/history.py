from __future__ import annotations

"""Compact history storage for summary snapshots."""

from array import array

import numpy as np
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
    """Append-only numeric store for summary snapshots."""

    def __init__(self) -> None:
        self._step = array("I")
        self._last_price = array("d")
        self._mid_price = array("d")
        self._best_bid = array("d")
        self._best_ask = array("d")
        self._spread = array("d")
        self._bid_depth = array("I")
        self._ask_depth = array("I")
        self._depth_imbalance = array("d")
        self._buy_aggr_volume = array("d")
        self._sell_aggr_volume = array("d")
        self._fair_price = array("d")
        self._cache: pd.DataFrame | None = None
        self._dirty = True

    def append(
        self,
        *,
        step: int,
        last_price: float,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        spread: float,
        bid_depth: int,
        ask_depth: int,
        depth_imbalance: float,
        buy_aggr_volume: float,
        sell_aggr_volume: float,
        fair_price: float,
    ) -> None:
        self._step.append(int(step))
        self._last_price.append(float(last_price))
        self._mid_price.append(float(mid_price))
        self._best_bid.append(float(best_bid))
        self._best_ask.append(float(best_ask))
        self._spread.append(float(spread))
        self._bid_depth.append(int(bid_depth))
        self._ask_depth.append(int(ask_depth))
        self._depth_imbalance.append(float(depth_imbalance))
        self._buy_aggr_volume.append(float(buy_aggr_volume))
        self._sell_aggr_volume.append(float(sell_aggr_volume))
        self._fair_price.append(float(fair_price))
        self._dirty = True

    def dataframe(self) -> pd.DataFrame:
        if self._cache is None or self._dirty:
            self._cache = pd.DataFrame(
                {
                    "step": np.frombuffer(self._step, dtype=np.uint32).astype(np.int64, copy=False),
                    "last_price": np.frombuffer(self._last_price, dtype=np.float64),
                    "mid_price": np.frombuffer(self._mid_price, dtype=np.float64),
                    "best_bid": np.frombuffer(self._best_bid, dtype=np.float64),
                    "best_ask": np.frombuffer(self._best_ask, dtype=np.float64),
                    "spread": np.frombuffer(self._spread, dtype=np.float64),
                    "bid_depth": np.frombuffer(self._bid_depth, dtype=np.uint32).astype(np.int64, copy=False),
                    "ask_depth": np.frombuffer(self._ask_depth, dtype=np.uint32).astype(np.int64, copy=False),
                    "depth_imbalance": np.frombuffer(self._depth_imbalance, dtype=np.float64),
                    "buy_aggr_volume": np.frombuffer(self._buy_aggr_volume, dtype=np.float64),
                    "sell_aggr_volume": np.frombuffer(self._sell_aggr_volume, dtype=np.float64),
                    "fair_price": np.frombuffer(self._fair_price, dtype=np.float64),
                },
                columns=SUMMARY_COLUMNS,
            )
            self._dirty = False
        return self._cache


__all__ = ["HistoryBuffer", "SUMMARY_COLUMNS"]
