from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict, TypeGuard

from .types import AggressorSide, ModelSide, ParticipantType

EventSide: TypeAlias = ModelSide | AggressorSide


class _DebugEventRecord(TypedDict):
    event_type: Literal["limit", "market", "cancel"]
    side: EventSide
    qty: float
    source: str
    participant_type: ParticipantType
    meta_order_id: int | None
    meta_order_side: AggressorSide | None


class LimitEvent(TypedDict):
    event_type: Literal["limit"]
    side: ModelSide
    qty: float
    source: str
    participant_type: ParticipantType
    meta_order_id: int | None
    meta_order_side: AggressorSide | None
    level: int


class MarketEvent(TypedDict):
    event_type: Literal["market"]
    side: AggressorSide
    qty: float
    source: str
    participant_type: ParticipantType
    meta_order_id: int | None
    meta_order_side: AggressorSide | None


class CancelEvent(TypedDict):
    event_type: Literal["cancel"]
    side: ModelSide
    qty: float
    source: str
    participant_type: ParticipantType
    meta_order_id: int | None
    meta_order_side: AggressorSide | None
    level: int | None
    tick: int


StepEvent = LimitEvent | MarketEvent | CancelEvent


class EventStateSnapshot(TypedDict):
    mid_price: float
    spread_ticks: float
    depth_imbalance: float
    best_bid_qty: float
    best_ask_qty: float


class EventLogRecord(TypedDict):
    event_type: Literal["limit", "market", "cancel"]
    side: EventSide
    level: int | None
    price: float
    requested_qty: float
    applied_qty: float
    fill_qty: float
    fills: tuple[tuple[float, float], ...]


__all__ = [
    "CancelEvent",
    "EventStateSnapshot",
    "EventLogRecord",
    "EventSide",
    "LimitEvent",
    "MarketEvent",
    "StepEvent",
    "build_debug_event_record",
    "is_cancel_event",
    "is_limit_event",
    "is_market_event",
    "make_cancel_event",
    "make_cancel_log_record",
    "make_limit_event",
    "make_limit_log_record",
    "make_market_event",
    "make_market_log_record",
]


def is_limit_event(event: StepEvent) -> TypeGuard[LimitEvent]:
    return event["event_type"] == "limit"


def is_market_event(event: StepEvent) -> TypeGuard[MarketEvent]:
    return event["event_type"] == "market"


def is_cancel_event(event: StepEvent) -> TypeGuard[CancelEvent]:
    return event["event_type"] == "cancel"


def make_limit_event(
    *,
    side: ModelSide,
    level: int,
    qty: float,
    source: str,
    participant_type: ParticipantType,
    meta_order_id: int | None,
    meta_order_side: AggressorSide | None,
) -> LimitEvent:
    return {
        "event_type": "limit",
        "side": side,
        "level": int(level),
        "qty": float(qty),
        "source": source,
        "participant_type": participant_type,
        "meta_order_id": meta_order_id,
        "meta_order_side": meta_order_side,
    }


def make_market_event(
    *,
    side: AggressorSide,
    qty: float,
    source: str,
    participant_type: ParticipantType,
    meta_order_id: int | None,
    meta_order_side: AggressorSide | None,
) -> MarketEvent:
    return {
        "event_type": "market",
        "side": side,
        "qty": float(qty),
        "source": source,
        "participant_type": participant_type,
        "meta_order_id": meta_order_id,
        "meta_order_side": meta_order_side,
    }


def make_cancel_event(
    *,
    side: ModelSide,
    level: int | None,
    tick: int,
    qty: float,
    source: str,
    participant_type: ParticipantType,
    meta_order_id: int | None,
    meta_order_side: AggressorSide | None,
) -> CancelEvent:
    return {
        "event_type": "cancel",
        "side": side,
        "level": None if level is None else int(level),
        "tick": int(tick),
        "qty": float(qty),
        "source": source,
        "participant_type": participant_type,
        "meta_order_id": meta_order_id,
        "meta_order_side": meta_order_side,
    }


def build_debug_event_record(event: StepEvent) -> _DebugEventRecord:
    return {
        "event_type": event["event_type"],
        "side": event["side"],
        "qty": float(event["qty"]),
        "source": event["source"],
        "participant_type": event["participant_type"],
        "meta_order_id": event["meta_order_id"],
        "meta_order_side": event["meta_order_side"],
    }


def make_limit_log_record(
    *,
    side: ModelSide,
    level: int,
    price: float,
    qty: float,
) -> EventLogRecord:
    return {
        "event_type": "limit",
        "side": side,
        "level": int(level),
        "price": float(price),
        "requested_qty": float(qty),
        "applied_qty": float(qty),
        "fill_qty": 0.0,
        "fills": (),
    }


def make_market_log_record(
    *,
    side: AggressorSide,
    price: float,
    requested_qty: float,
    applied_qty: float,
    fills: tuple[tuple[float, float], ...],
) -> EventLogRecord:
    return {
        "event_type": "market",
        "side": side,
        "level": None,
        "price": float(price),
        "requested_qty": float(requested_qty),
        "applied_qty": float(applied_qty),
        "fill_qty": float(applied_qty),
        "fills": fills,
    }


def make_cancel_log_record(
    *,
    side: ModelSide,
    level: int | None,
    price: float,
    requested_qty: float,
    applied_qty: float,
) -> EventLogRecord:
    return {
        "event_type": "cancel",
        "side": side,
        "level": None if level is None else int(level),
        "price": float(price),
        "requested_qty": float(requested_qty),
        "applied_qty": float(applied_qty),
        "fill_qty": 0.0,
        "fills": (),
    }
