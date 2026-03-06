from __future__ import annotations

import copy

import pandas as pd


SUMMARY_COLUMNS = [
    "step",
    "last_price",
    "mid_price",
    "microprice",
    "best_bid",
    "best_ask",
    "spread",
    "buy_aggr_volume",
    "sell_aggr_volume",
    "trade_strength",
    "depth_imbalance",
    "regime",
    "top_n_bid_qty",
    "top_n_ask_qty",
    "realized_vol",
    "signed_flow",
]


class HistoryBuffer:
    def __init__(self) -> None:
        self._current_snapshot: dict[str, object] | None = None
        self._rows: list[dict[str, object]] = []

    def record(
        self,
        snapshot: dict[str, object],
        *,
        top_n_bid_qty: float,
        top_n_ask_qty: float,
        realized_vol: float,
        signed_flow: float,
    ) -> None:
        self._current_snapshot = copy.deepcopy(snapshot)
        row = {
            "step": snapshot["step"],
            "last_price": snapshot["last_price"],
            "mid_price": snapshot["mid_price"],
            "microprice": snapshot["microprice"],
            "best_bid": snapshot["best_bid"],
            "best_ask": snapshot["best_ask"],
            "spread": snapshot["spread"],
            "buy_aggr_volume": snapshot["buy_aggr_volume"],
            "sell_aggr_volume": snapshot["sell_aggr_volume"],
            "trade_strength": snapshot["trade_strength"],
            "depth_imbalance": snapshot["depth_imbalance"],
            "regime": snapshot["regime"],
            "top_n_bid_qty": top_n_bid_qty,
            "top_n_ask_qty": top_n_ask_qty,
            "realized_vol": realized_vol,
            "signed_flow": signed_flow,
        }
        self._rows.append(row)

    def current(self) -> dict[str, object]:
        if self._current_snapshot is None:
            raise ValueError("no snapshot recorded")
        return copy.deepcopy(self._current_snapshot)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=SUMMARY_COLUMNS).copy(deep=True)
