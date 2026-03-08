from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from orderwave import Market


def test_event_history_is_reproducible_for_same_seed() -> None:
    market_a = Market(seed=21, config={"preset": "balanced"})
    market_b = Market(seed=21, config={"preset": "balanced"})

    market_a.gen(steps=40)
    market_b.gen(steps=40)

    pd.testing.assert_frame_equal(market_a.get_event_history(), market_b.get_event_history())
    pd.testing.assert_frame_equal(market_a.get_debug_history(), market_b.get_debug_history())


def test_event_history_columns_and_invariants_hold() -> None:
    market = Market(seed=42, config={"preset": "trend", "market_rate_scale": 1.2})
    market.gen(steps=120)

    events = market.get_event_history()
    debug = market.get_debug_history()

    expected = {
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
    }

    assert set(events.columns) == expected
    assert not events.empty
    assert len(events) == len(debug)
    assert (events["best_bid_after"] < events["best_ask_after"]).all()

    ordered = events[["step", "event_idx"]].to_records(index=False).tolist()
    assert ordered == sorted(ordered)

    market_rows = events.loc[events["event_type"] == "market"]
    assert not market_rows.empty
    for row in market_rows.itertuples(index=False):
        assert row.fill_qty == pytest.approx(sum(qty for _, qty in row.fills))

    non_market_rows = events.loc[events["event_type"] != "market"]
    assert non_market_rows["fills"].apply(lambda fills: fills == []).all()


def test_debug_history_aligns_one_to_one_with_event_history() -> None:
    market = Market(seed=17, config={"preset": "balanced"})
    market.gen(steps=80)

    events = market.get_event_history()
    debug = market.get_debug_history()
    joined = events.merge(debug, on=["step", "event_idx"], how="inner")

    assert len(joined) == len(events)
    assert joined["session_phase_x"].equals(joined["session_phase_y"])
    assert set(debug["session_phase"].unique()) <= {"open", "mid", "close"}


def test_trade_strength_matches_execution_only_ema() -> None:
    market = Market(seed=7, config={"preset": "volatile"})
    market.gen(steps=100)

    history = market.get_history()
    events = market.get_event_history()
    market_events = events.loc[events["event_type"] == "market"]
    buy_by_step = market_events.loc[market_events["side"] == "buy"].groupby("step")["fill_qty"].sum()
    sell_by_step = market_events.loc[market_events["side"] == "sell"].groupby("step")["fill_qty"].sum()

    alpha = 2.0 / (market.config.flow_window + 1.0)
    buy_ema = 0.0
    sell_ema = 0.0
    expected = []
    for step in history["step"]:
        if step == 0:
            expected.append(0.0)
            continue
        buy_ema = ((1.0 - alpha) * buy_ema) + (alpha * float(buy_by_step.get(step, 0.0)))
        sell_ema = ((1.0 - alpha) * sell_ema) + (alpha * float(sell_by_step.get(step, 0.0)))
        expected.append((buy_ema - sell_ema) / max(buy_ema + sell_ema, 1e-9))

    np.testing.assert_allclose(history["trade_strength"].to_numpy(dtype=float), np.array(expected, dtype=float))


def test_balanced_preset_statistical_guardrails_hold() -> None:
    market = Market(seed=42, config={"preset": "balanced"})
    market.gen(steps=2000)

    history = market.get_history()
    events = market.get_event_history()
    debug = market.get_debug_history()
    mid_ret = history["mid_price"].diff().fillna(0.0)
    next_mid_ret = mid_ret.shift(-1).fillna(0.0)
    corr = float(history["depth_imbalance"].corr(next_mid_ret))
    abs_ret = mid_ret.abs()
    nonzero_abs_ret = abs_ret.loc[abs_ret > 0.0]
    acf_1 = float(nonzero_abs_ret.autocorr(lag=1))
    market_events = events.loc[events["event_type"] == "market"].copy()
    buy_count_acf = float(
        market_events.assign(is_buy=(market_events["side"] == "buy").astype(float))
        .groupby("step")["is_buy"]
        .sum()
        .autocorr(lag=1)
    )
    cancel_count_acf = float(
        events.assign(is_cancel=(events["event_type"] == "cancel").astype(float))
        .groupby("step")["is_cancel"]
        .sum()
        .autocorr(lag=1)
    )
    phase_fill = events.groupby(["session_phase", "step"])["fill_qty"].sum().groupby("session_phase").mean()
    phase_spread = history.groupby("session_phase")["spread"].mean()
    joined = events.merge(debug, on=["step", "event_idx"], how="inner")
    market_joined = joined.loc[joined["event_type"] == "market"].copy()
    market_joined["sign"] = market_joined["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)
    meta_step_sign = market_joined.loc[market_joined["meta_order_id"].notna()].groupby("step")["sign"].sum()
    meta_directionality = float(meta_step_sign.abs().mean())

    step_shock = debug.groupby("step")["shock_state"].agg(
        lambda states: "none" if (states == "none").all() else next(value for value in states if value != "none")
    )
    step_view = history.set_index("step").join(step_shock.rename("shock_state")).fillna({"shock_state": "none"})
    shock_steps = step_view.loc[step_view["shock_state"] != "none"]
    calm_steps = step_view.loc[step_view["shock_state"] == "none"]
    shock_abs_ret = float(shock_steps["mid_price"].diff().abs().mean())
    calm_abs_ret = float(calm_steps["mid_price"].diff().abs().mean())
    first_500_events = events.loc[events["step"] <= 500]
    first_500_market = first_500_events.loc[first_500_events["event_type"] == "market"]
    market_buy_count = int((first_500_market["side"] == "buy").sum())
    market_sell_count = int((first_500_market["side"] == "sell").sum())
    market_buy_share = market_buy_count / max(market_buy_count + market_sell_count, 1)
    events_per_step = len(events) / 2000.0

    assert 0.0 < corr < 0.25
    assert acf_1 > 0.0
    assert history["spread"].nunique() >= 3
    assert events_per_step < 40.0
    assert history["session_phase"].nunique() == 3
    assert (phase_fill.max() - phase_fill.min()) > 2.0
    assert (phase_spread.max() - phase_spread.min()) > 0.001
    assert 0.35 < market_buy_share < 0.65
    assert market_buy_count > 100
    assert market_sell_count > 100
    assert buy_count_acf > 0.05
    assert cancel_count_acf > 0.05
    assert meta_directionality >= 1.0
    assert shock_abs_ret > calm_abs_ret
