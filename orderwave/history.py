from __future__ import annotations

from array import array
from typing import Any, Sequence

import numpy as np
import pandas as pd

from orderwave.visualization import VisualHistoryStore


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

_PHASES = ("open", "mid", "close")
_REGIMES = ("calm", "directional", "stressed")
_EVENT_TYPES = ("limit", "market", "cancel")
_EVENT_SIDES = ("bid", "ask", "buy", "sell")
_INT_SENTINEL = -(2**31)


class _CategoryCodec:
    def __init__(self, initial_labels: Sequence[str] = ()) -> None:
        self._labels: list[str | None] = [None]
        self._codes: dict[str | None, int] = {None: 0}
        for label in initial_labels:
            self.encode(label)

    def encode(self, value: str | None) -> int:
        key = None if value is None else str(value)
        code = self._codes.get(key)
        if code is not None:
            return code
        code = len(self._labels)
        self._labels.append(key)
        self._codes[key] = code
        return code

    def decode_many(self, codes: array) -> list[str | None]:
        labels = self._labels
        return [labels[int(code)] if 0 <= int(code) < len(labels) else None for code in codes]


def _nullable_int_series(values: array, *, sentinel: int = _INT_SENTINEL) -> pd.Series:
    if not values:
        return pd.Series(pd.array([], dtype="Int32"))
    materialized = [pd.NA if int(value) == sentinel else int(value) for value in values]
    return pd.Series(pd.array(materialized, dtype="Int32"))


class _SummaryStore:
    def __init__(self) -> None:
        self._step = array("I")
        self._day = array("I")
        self._session_step = array("I")
        self._session_phase = array("B")
        self._last_price = array("d")
        self._mid_price = array("d")
        self._microprice = array("d")
        self._best_bid = array("d")
        self._best_ask = array("d")
        self._spread = array("d")
        self._buy_aggr_volume = array("d")
        self._sell_aggr_volume = array("d")
        self._trade_strength = array("d")
        self._depth_imbalance = array("d")
        self._regime = array("B")
        self._top_n_bid_qty = array("d")
        self._top_n_ask_qty = array("d")
        self._realized_vol = array("d")
        self._signed_flow = array("d")
        self._phase_codec = _CategoryCodec(_PHASES)
        self._regime_codec = _CategoryCodec(_REGIMES)
        self._cache: pd.DataFrame | None = None
        self._dirty = True

    def append(
        self,
        snapshot: dict[str, object],
        *,
        top_n_bid_qty: float,
        top_n_ask_qty: float,
        realized_vol: float,
        signed_flow: float,
    ) -> None:
        self._step.append(int(snapshot["step"]))
        self._day.append(int(snapshot["day"]))
        self._session_step.append(int(snapshot["session_step"]))
        self._session_phase.append(self._phase_codec.encode(str(snapshot["session_phase"])))
        self._last_price.append(float(snapshot["last_price"]))
        self._mid_price.append(float(snapshot["mid_price"]))
        self._microprice.append(float(snapshot["microprice"]))
        self._best_bid.append(float(snapshot["best_bid"]))
        self._best_ask.append(float(snapshot["best_ask"]))
        self._spread.append(float(snapshot["spread"]))
        self._buy_aggr_volume.append(float(snapshot["buy_aggr_volume"]))
        self._sell_aggr_volume.append(float(snapshot["sell_aggr_volume"]))
        self._trade_strength.append(float(snapshot["trade_strength"]))
        self._depth_imbalance.append(float(snapshot["depth_imbalance"]))
        self._regime.append(self._regime_codec.encode(str(snapshot["regime"])))
        self._top_n_bid_qty.append(float(top_n_bid_qty))
        self._top_n_ask_qty.append(float(top_n_ask_qty))
        self._realized_vol.append(float(realized_vol))
        self._signed_flow.append(float(signed_flow))
        self._dirty = True

    def dataframe(self) -> pd.DataFrame:
        if self._cache is None or self._dirty:
            self._cache = pd.DataFrame(
                {
                    "step": np.frombuffer(self._step, dtype=np.uint32).astype(np.int64, copy=False),
                    "day": np.frombuffer(self._day, dtype=np.uint32).astype(np.int64, copy=False),
                    "session_step": np.frombuffer(self._session_step, dtype=np.uint32).astype(np.int64, copy=False),
                    "session_phase": self._phase_codec.decode_many(self._session_phase),
                    "last_price": np.frombuffer(self._last_price, dtype=np.float64),
                    "mid_price": np.frombuffer(self._mid_price, dtype=np.float64),
                    "microprice": np.frombuffer(self._microprice, dtype=np.float64),
                    "best_bid": np.frombuffer(self._best_bid, dtype=np.float64),
                    "best_ask": np.frombuffer(self._best_ask, dtype=np.float64),
                    "spread": np.frombuffer(self._spread, dtype=np.float64),
                    "buy_aggr_volume": np.frombuffer(self._buy_aggr_volume, dtype=np.float64),
                    "sell_aggr_volume": np.frombuffer(self._sell_aggr_volume, dtype=np.float64),
                    "trade_strength": np.frombuffer(self._trade_strength, dtype=np.float64),
                    "depth_imbalance": np.frombuffer(self._depth_imbalance, dtype=np.float64),
                    "regime": self._regime_codec.decode_many(self._regime),
                    "top_n_bid_qty": np.frombuffer(self._top_n_bid_qty, dtype=np.float64),
                    "top_n_ask_qty": np.frombuffer(self._top_n_ask_qty, dtype=np.float64),
                    "realized_vol": np.frombuffer(self._realized_vol, dtype=np.float64),
                    "signed_flow": np.frombuffer(self._signed_flow, dtype=np.float64),
                },
                columns=SUMMARY_COLUMNS,
            )
            self._dirty = False
        return self._cache


class _EventStore:
    def __init__(self) -> None:
        self._step = array("I")
        self._event_idx = array("I")
        self._day = array("I")
        self._session_step = array("I")
        self._session_phase = array("B")
        self._event_type = array("B")
        self._side = array("B")
        self._level = array("i")
        self._price = array("d")
        self._requested_qty = array("d")
        self._applied_qty = array("d")
        self._fill_qty = array("d")
        self._best_bid_after = array("d")
        self._best_ask_after = array("d")
        self._mid_price_after = array("d")
        self._last_trade_price_after = array("d")
        self._regime = array("B")
        self._fill_start = array("I")
        self._fill_count = array("I")
        self._flat_fill_price = array("d")
        self._flat_fill_qty = array("d")
        self._phase_codec = _CategoryCodec(_PHASES)
        self._event_type_codec = _CategoryCodec(_EVENT_TYPES)
        self._side_codec = _CategoryCodec(_EVENT_SIDES)
        self._regime_codec = _CategoryCodec(_REGIMES)
        self._cache: pd.DataFrame | None = None
        self._dirty = True

    def append(
        self,
        step: int,
        event_idx: int,
        day: int,
        session_step: int,
        session_phase: str,
        event_type: str,
        side: str,
        level: int | None,
        price: float,
        requested_qty: float,
        applied_qty: float,
        fill_qty: float,
        fills: Sequence[tuple[float, float]],
        best_bid_after: float,
        best_ask_after: float,
        mid_price_after: float,
        last_trade_price_after: float,
        regime: str,
    ) -> None:
        self._step.append(int(step))
        self._event_idx.append(int(event_idx))
        self._day.append(int(day))
        self._session_step.append(int(session_step))
        self._session_phase.append(self._phase_codec.encode(session_phase))
        self._event_type.append(self._event_type_codec.encode(event_type))
        self._side.append(self._side_codec.encode(side))
        self._level.append(_INT_SENTINEL if level is None else int(level))
        self._price.append(float(price))
        self._requested_qty.append(float(requested_qty))
        self._applied_qty.append(float(applied_qty))
        self._fill_qty.append(float(fill_qty))
        self._best_bid_after.append(float(best_bid_after))
        self._best_ask_after.append(float(best_ask_after))
        self._mid_price_after.append(float(mid_price_after))
        self._last_trade_price_after.append(float(last_trade_price_after))
        self._regime.append(self._regime_codec.encode(regime))
        self._fill_start.append(len(self._flat_fill_price))
        self._fill_count.append(len(fills))
        for fill_price, fill_size in fills:
            self._flat_fill_price.append(float(fill_price))
            self._flat_fill_qty.append(float(fill_size))
        self._dirty = True

    def dataframe(self) -> pd.DataFrame:
        if self._cache is None or self._dirty:
            frame = pd.DataFrame(
                {
                    "step": np.frombuffer(self._step, dtype=np.uint32).astype(np.int64, copy=False),
                    "event_idx": np.frombuffer(self._event_idx, dtype=np.uint32).astype(np.int64, copy=False),
                    "day": np.frombuffer(self._day, dtype=np.uint32).astype(np.int64, copy=False),
                    "session_step": np.frombuffer(self._session_step, dtype=np.uint32).astype(np.int64, copy=False),
                    "session_phase": self._phase_codec.decode_many(self._session_phase),
                    "event_type": self._event_type_codec.decode_many(self._event_type),
                    "side": self._side_codec.decode_many(self._side),
                    "level": _nullable_int_series(self._level),
                    "price": np.frombuffer(self._price, dtype=np.float64),
                    "requested_qty": np.frombuffer(self._requested_qty, dtype=np.float64),
                    "applied_qty": np.frombuffer(self._applied_qty, dtype=np.float64),
                    "fill_qty": np.frombuffer(self._fill_qty, dtype=np.float64),
                    "fills": self._materialize_fills(),
                    "best_bid_after": np.frombuffer(self._best_bid_after, dtype=np.float64),
                    "best_ask_after": np.frombuffer(self._best_ask_after, dtype=np.float64),
                    "mid_price_after": np.frombuffer(self._mid_price_after, dtype=np.float64),
                    "last_trade_price_after": np.frombuffer(self._last_trade_price_after, dtype=np.float64),
                    "regime": self._regime_codec.decode_many(self._regime),
                },
                columns=EVENT_COLUMNS,
            )
            self._cache = frame
            self._dirty = False
        return self._cache

    def _materialize_fills(self) -> list[list[tuple[float, float]]]:
        fills: list[list[tuple[float, float]]] = []
        flat_price = self._flat_fill_price
        flat_qty = self._flat_fill_qty
        for start, count in zip(self._fill_start, self._fill_count):
            begin = int(start)
            end = begin + int(count)
            fills.append([(float(flat_price[index]), float(flat_qty[index])) for index in range(begin, end)])
        return fills


class _DebugStore:
    def __init__(self) -> None:
        self._step = array("I")
        self._event_idx = array("I")
        self._day = array("I")
        self._session_step = array("I")
        self._session_phase = array("B")
        self._source = array("H")
        self._participant_type = array("H")
        self._meta_order_id = array("i")
        self._meta_order_side = array("H")
        self._meta_order_progress = array("d")
        self._burst_state = array("H")
        self._shock_state = array("H")
        self._phase_codec = _CategoryCodec(_PHASES)
        self._source_codec = _CategoryCodec()
        self._participant_codec = _CategoryCodec()
        self._meta_side_codec = _CategoryCodec(("buy", "sell"))
        self._burst_codec = _CategoryCodec()
        self._shock_codec = _CategoryCodec()
        self._cache: pd.DataFrame | None = None
        self._dirty = True

    def append(
        self,
        step: int,
        event_idx: int,
        day: int,
        session_step: int,
        session_phase: str,
        source: str | None,
        participant_type: str | None,
        meta_order_id: int | None,
        meta_order_side: str | None,
        meta_order_progress: float | None,
        burst_state: str | None,
        shock_state: str | None,
    ) -> None:
        self._step.append(int(step))
        self._event_idx.append(int(event_idx))
        self._day.append(int(day))
        self._session_step.append(int(session_step))
        self._session_phase.append(self._phase_codec.encode(session_phase))
        self._source.append(self._source_codec.encode(source))
        self._participant_type.append(self._participant_codec.encode(participant_type))
        self._meta_order_id.append(_INT_SENTINEL if meta_order_id is None else int(meta_order_id))
        self._meta_order_side.append(self._meta_side_codec.encode(meta_order_side))
        self._meta_order_progress.append(float("nan") if meta_order_progress is None else float(meta_order_progress))
        self._burst_state.append(self._burst_codec.encode(burst_state))
        self._shock_state.append(self._shock_codec.encode(shock_state))
        self._dirty = True

    def dataframe(self) -> pd.DataFrame:
        if self._cache is None or self._dirty:
            frame = pd.DataFrame(
                {
                    "step": np.frombuffer(self._step, dtype=np.uint32).astype(np.int64, copy=False),
                    "event_idx": np.frombuffer(self._event_idx, dtype=np.uint32).astype(np.int64, copy=False),
                    "day": np.frombuffer(self._day, dtype=np.uint32).astype(np.int64, copy=False),
                    "session_step": np.frombuffer(self._session_step, dtype=np.uint32).astype(np.int64, copy=False),
                    "session_phase": self._phase_codec.decode_many(self._session_phase),
                    "source": self._source_codec.decode_many(self._source),
                    "participant_type": self._participant_codec.decode_many(self._participant_type),
                    "meta_order_id": _nullable_int_series(self._meta_order_id),
                    "meta_order_side": self._meta_side_codec.decode_many(self._meta_order_side),
                    "meta_order_progress": np.frombuffer(self._meta_order_progress, dtype=np.float64),
                    "burst_state": self._burst_codec.decode_many(self._burst_state),
                    "shock_state": self._shock_codec.decode_many(self._shock_state),
                },
                columns=DEBUG_COLUMNS,
            )
            self._cache = frame
            self._dirty = False
        return self._cache


class HistoryBuffer:
    def __init__(self, *, logging_mode: str, visual_depth: int) -> None:
        self._logging_mode = logging_mode
        self._current_snapshot: dict[str, object] | None = None
        self._summary = _SummaryStore()
        self._events = _EventStore() if logging_mode == "full" else None
        self._debug = _DebugStore() if logging_mode == "full" else None
        self.visual_store = VisualHistoryStore(depth=visual_depth)

    @property
    def logging_mode(self) -> str:
        return self._logging_mode

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
        self._summary.append(
            snapshot,
            top_n_bid_qty=top_n_bid_qty,
            top_n_ask_qty=top_n_ask_qty,
            realized_vol=realized_vol,
            signed_flow=signed_flow,
        )

    def record_event(
        self,
        step: int,
        event_idx: int,
        day: int,
        session_step: int,
        session_phase: str,
        event_type: str,
        side: str,
        level: int | None,
        price: float,
        requested_qty: float,
        applied_qty: float,
        fill_qty: float,
        fills: Sequence[tuple[float, float]],
        best_bid_after: float,
        best_ask_after: float,
        mid_price_after: float,
        last_trade_price_after: float,
        regime: str,
    ) -> None:
        if self._events is None:
            return
        self._events.append(
            step,
            event_idx,
            day,
            session_step,
            session_phase,
            event_type,
            side,
            level,
            price,
            requested_qty,
            applied_qty,
            fill_qty,
            fills,
            best_bid_after,
            best_ask_after,
            mid_price_after,
            last_trade_price_after,
            regime,
        )

    def record_debug(
        self,
        step: int,
        event_idx: int,
        day: int,
        session_step: int,
        session_phase: str,
        source: str | None,
        participant_type: str | None,
        meta_order_id: int | None,
        meta_order_side: str | None,
        meta_order_progress: float | None,
        burst_state: str | None,
        shock_state: str | None,
    ) -> None:
        if self._debug is None:
            return
        self._debug.append(
            step,
            event_idx,
            day,
            session_step,
            session_phase,
            source,
            participant_type,
            meta_order_id,
            meta_order_side,
            meta_order_progress,
            burst_state,
            shock_state,
        )

    def record_visual(
        self,
        *,
        step: int,
        bid_levels: Sequence[tuple[int, int]],
        ask_levels: Sequence[tuple[int, int]],
    ) -> None:
        self.visual_store.append(step=step, bid_levels=bid_levels, ask_levels=ask_levels)

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
        return self._summary.dataframe()

    def event_dataframe(self) -> pd.DataFrame:
        if self._events is None:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        return self._events.dataframe()

    def debug_dataframe(self) -> pd.DataFrame:
        if self._debug is None:
            return pd.DataFrame(columns=DEBUG_COLUMNS)
        return self._debug.dataframe()
