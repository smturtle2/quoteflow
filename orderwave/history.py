from __future__ import annotations

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
        self._rows: list[tuple[object, ...]] = []
        self._event_rows: list[tuple[object, ...]] = []
        self._debug_rows: list[tuple[object, ...]] = []

    def record(
        self,
        snapshot: dict[str, object],
        *,
        top_n_bid_qty: float,
        top_n_ask_qty: float,
        realized_vol: float,
        signed_flow: float,
    ) -> None:
        self._current_snapshot = snapshot
        self._rows.append(
            (
                snapshot["step"],
                snapshot["day"],
                snapshot["session_step"],
                snapshot["session_phase"],
                snapshot["last_price"],
                snapshot["mid_price"],
                snapshot["microprice"],
                snapshot["best_bid"],
                snapshot["best_ask"],
                snapshot["spread"],
                snapshot["buy_aggr_volume"],
                snapshot["sell_aggr_volume"],
                snapshot["trade_strength"],
                snapshot["depth_imbalance"],
                snapshot["regime"],
                top_n_bid_qty,
                top_n_ask_qty,
                realized_vol,
                signed_flow,
            )
        )

    def record_event(self, event_row: tuple[object, ...] | dict[str, object]) -> None:
        if isinstance(event_row, tuple):
            self._event_rows.append(event_row)
            return
        self._event_rows.append(tuple(event_row[column] for column in EVENT_COLUMNS))

    def record_debug(self, debug_row: tuple[object, ...] | dict[str, object]) -> None:
        if isinstance(debug_row, tuple):
            self._debug_rows.append(debug_row)
            return
        self._debug_rows.append(tuple(debug_row[column] for column in DEBUG_COLUMNS))

    def current(self) -> dict[str, object]:
        if self._current_snapshot is None:
            raise ValueError("no snapshot recorded")
        snapshot = self._current_snapshot
        return {
            **snapshot,
            "bids": [dict(level) for level in snapshot["bids"]],
            "asks": [dict(level) for level in snapshot["asks"]],
        }

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._rows, columns=SUMMARY_COLUMNS)

    def event_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._event_rows, columns=EVENT_COLUMNS)

    def debug_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._debug_rows, columns=DEBUG_COLUMNS)
