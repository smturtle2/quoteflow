from __future__ import annotations

from orderwave._model.events import (
    build_debug_event_record,
    is_cancel_event,
    is_limit_event,
    is_market_event,
    make_cancel_event,
    make_cancel_log_record,
    make_limit_event,
    make_limit_log_record,
    make_market_event,
    make_market_log_record,
)


def test_event_constructors_and_narrowing_helpers_cover_all_step_event_shapes() -> None:
    limit_event = make_limit_event(
        side="bid",
        level=1,
        qty=5.0,
        source="test",
        participant_type="passive_lp",
        meta_order_id=None,
        meta_order_side=None,
    )
    market_event = make_market_event(
        side="buy",
        qty=7.5,
        source="test",
        participant_type="noise_taker",
        meta_order_id=3,
        meta_order_side="buy",
    )
    cancel_event = make_cancel_event(
        side="ask",
        level=2,
        tick=10005,
        qty=4.0,
        source="test",
        participant_type="inventory_mm",
        meta_order_id=4,
        meta_order_side="sell",
    )

    assert is_limit_event(limit_event)
    assert not is_market_event(limit_event)
    assert not is_cancel_event(limit_event)
    assert limit_event["level"] == 1

    assert is_market_event(market_event)
    assert not is_limit_event(market_event)
    assert not is_cancel_event(market_event)
    assert market_event["side"] == "buy"

    assert is_cancel_event(cancel_event)
    assert not is_limit_event(cancel_event)
    assert not is_market_event(cancel_event)
    assert cancel_event["tick"] == 10005


def test_build_debug_event_record_keeps_common_event_payload_fields() -> None:
    event = make_market_event(
        side="sell",
        qty=3.25,
        source="sampler",
        participant_type="informed_meta",
        meta_order_id=11,
        meta_order_side="sell",
    )

    record = build_debug_event_record(event)

    assert record == {
        "event_type": "market",
        "side": "sell",
        "qty": 3.25,
        "source": "sampler",
        "participant_type": "informed_meta",
        "meta_order_id": 11,
        "meta_order_side": "sell",
    }


def test_log_record_helpers_keep_event_history_payload_typed_and_consistent() -> None:
    limit_record = make_limit_log_record(side="bid", level=0, price=100.0, qty=5.0)
    market_record = make_market_log_record(
        side="buy",
        price=100.02,
        requested_qty=7.0,
        applied_qty=6.0,
        fills=((100.01, 2.0), (100.02, 4.0)),
    )
    cancel_record = make_cancel_log_record(
        side="ask",
        level=2,
        price=100.05,
        requested_qty=4.0,
        applied_qty=3.0,
    )

    assert limit_record["event_type"] == "limit"
    assert limit_record["fill_qty"] == 0.0
    assert limit_record["fills"] == ()

    assert market_record["event_type"] == "market"
    assert market_record["fill_qty"] == 6.0
    assert sum(qty for _, qty in market_record["fills"]) == 6.0

    assert cancel_record["event_type"] == "cancel"
    assert cancel_record["level"] == 2
    assert cancel_record["fill_qty"] == 0.0
