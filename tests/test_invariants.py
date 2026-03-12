from __future__ import annotations

from orderwave import Market


def test_book_invariants_hold_over_long_run() -> None:
    market = Market(seed=7, config={"max_spread_ticks": 4, "max_fair_move_ticks": 2})
    history = market.run(steps=400).history

    assert bool((history["best_bid"] < history["best_ask"]).all())
    assert bool((history["bid_depth"] >= 0).all())
    assert bool((history["ask_depth"] >= 0).all())
    assert float(history["spread"].max()) <= (4 * market.tick_size) + 1e-9

    fair_gap = (history["fair_price"] - history["mid_price"]).abs() / market.tick_size
    assert float(fair_gap.max()) <= 4.0 + 1e-9


def test_quantity_sampler_respects_config_bounds() -> None:
    market = Market(seed=19, config={"min_order_qty": 2, "max_order_qty": 4})
    samples = [market._sample_quantity() for _ in range(500)]

    assert min(samples) >= 2
    assert max(samples) <= 4
