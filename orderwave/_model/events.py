from __future__ import annotations

from typing import Literal, TypedDict

from .types import AggressorSide, ModelSide, ParticipantType


EventSide = ModelSide | AggressorSide


class _BaseEvent(TypedDict):
    event_type: Literal["limit", "market", "cancel"]
    side: EventSide
    qty: float
    source: str
    participant_type: ParticipantType
    meta_order_id: int | None
    meta_order_side: AggressorSide | None


class LimitEvent(_BaseEvent):
    event_type: Literal["limit"]
    side: ModelSide
    level: int


class MarketEvent(_BaseEvent):
    event_type: Literal["market"]
    side: AggressorSide


class CancelEvent(_BaseEvent):
    event_type: Literal["cancel"]
    side: ModelSide
    level: int | None
    tick: int


StepEvent = LimitEvent | MarketEvent | CancelEvent


class AppliedEventResult(TypedDict):
    side: EventSide
    fill_qty: float


class EventLogRecord(TypedDict):
    event_type: Literal["limit", "market", "cancel"]
    side: EventSide
    level: int | None
    price: float
    requested_qty: float
    applied_qty: float
    fill_qty: float
    fills: tuple[tuple[float, float], ...]


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


def build_debug_event_record(event: StepEvent) -> _BaseEvent:
    return {
        "event_type": event["event_type"],
        "side": event["side"],
        "qty": float(event["qty"]),
        "source": event["source"],
        "participant_type": event["participant_type"],
        "meta_order_id": event["meta_order_id"],
        "meta_order_side": event["meta_order_side"],
    }
