from __future__ import annotations

import copy

import pandas as pd


SUMMARY_COLUMNS = [
    "step",
    "day",
    "session_step",
    "session_phase",
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

EVENT_COLUMNS = [
    "step",
    "event_idx",
    "day",
    "session_step",
    "session_phase",
    "event_type",
    "side",
    "level",
    "price",
    "requested_qty",
    "applied_qty",
    "fill_qty",
    "fills",
    "best_bid_after",
    "best_ask_after",
    "mid_price_after",
    "last_trade_price_after",
    "regime",
]

DEBUG_COLUMNS = [
    "step",
    "event_idx",
    "day",
    "session_step",
    "session_phase",
    "source",
    "participant_type",
    "meta_order_id",
    "meta_order_side",
    "meta_order_progress",
    "burst_state",
    "shock_state",
]


class HistoryBuffer:
    def __init__(self) -> None:
        self._current_snapshot: dict[str, object] | None = None
        self._rows: list[dict[str, object]] = []
        self._event_rows: list[dict[str, object]] = []
        self._debug_rows: list[dict[str, object]] = []

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
            "day": snapshot["day"],
            "session_step": snapshot["session_step"],
            "session_phase": snapshot["session_phase"],
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

    def record_event(self, event_row: dict[str, object]) -> None:
        self._event_rows.append(copy.deepcopy(event_row))

    def record_debug(self, debug_row: dict[str, object]) -> None:
        self._debug_rows.append(copy.deepcopy(debug_row))

    def current(self) -> dict[str, object]:
        if self._current_snapshot is None:
            raise ValueError("no snapshot recorded")
        return copy.deepcopy(self._current_snapshot)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=SUMMARY_COLUMNS).copy(deep=True)

    def event_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._event_rows, columns=EVENT_COLUMNS).copy(deep=True)

    def debug_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._debug_rows, columns=DEBUG_COLUMNS).copy(deep=True)
