from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .shared import (
    EPSILON,
    PHASE_ORDER,
    ValidationRun,
    directional_ratio,
    safe_autocorr,
    safe_corr,
    safe_mean,
    safe_std,
    tick_size_from_run,
)


def compute_run_metrics(
    run: ValidationRun,
    *,
    stage: str,
    preset: str,
    seed: int,
    config_label: str = "baseline",
    knob_name: str | None = None,
    knob_scale: float | None = None,
    repeat_idx: int | None = None,
) -> dict[str, Any]:
    """Compute validation metrics for one completed run."""

    step_index = analysis_step_index(run)
    history = analysis_history(run)
    events = analysis_events(run)
    debug = analysis_debug(run)
    if "event_type" not in events.columns:
        events = pd.DataFrame(columns=["event_type", "side", "step", "session_phase", "fill_qty"])
    if "shock_state" not in debug.columns:
        debug = pd.DataFrame(columns=["step", "shock_state", "meta_order_id"])
    event_count = len(events)
    effective_steps = max(len(step_index), 1)
    logged_event_count = len(run.event_history) + len(run.debug_history)

    mid_return = history["mid_price"].diff().fillna(0.0) if not history.empty else pd.Series(dtype=float)
    next_mid_return = mid_return.shift(-1).fillna(0.0) if not mid_return.empty else pd.Series(dtype=float)
    abs_return = mid_return.abs() if not mid_return.empty else pd.Series(dtype=float)

    market_events = events.loc[events["event_type"] == "market"].copy() if "event_type" in events else pd.DataFrame()
    limit_events = events.loc[events["event_type"] == "limit"].copy() if "event_type" in events else pd.DataFrame()
    cancel_events = events.loc[events["event_type"] == "cancel"].copy() if "event_type" in events else pd.DataFrame()

    buy_count = int((market_events["side"] == "buy").sum())
    sell_count = int((market_events["side"] == "sell").sum())
    market_event_count = buy_count + sell_count
    market_signs = (
        market_events["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)
        if market_event_count
        else pd.Series(dtype=float)
    )

    phase_metrics = phase_metrics_by_session(history, events, market_events)
    joined = join_events_and_debug(events, debug)
    shock_abs_return, calm_abs_return = shock_vs_calm_abs_return(history, debug)
    meta_active_ratio, meta_inactive_ratio = meta_directional_ratios(joined)

    crossing_history = int((run.history["best_bid"] >= run.history["best_ask"]).sum()) if not run.history.empty else 0
    crossing_events = (
        int((run.event_history["best_bid_after"] >= run.event_history["best_ask_after"]).sum())
        if not run.event_history.empty
        else 0
    )
    realized_vol = safe_mean(history["realized_vol"]) if not history.empty else 0.0
    if realized_vol <= 0.0:
        realized_vol = safe_std(mid_return)
    spread_q90 = _series_quantile(history["spread"], 0.90) if not history.empty else 0.0
    spread_q99 = _series_quantile(history["spread"], 0.99) if not history.empty else 0.0
    abs_return_q90 = _series_quantile(abs_return, 0.90) if not abs_return.empty else 0.0
    abs_return_q99 = _series_quantile(abs_return, 0.99) if not abs_return.empty else 0.0
    visible_levels_bid = safe_mean(history["visible_levels_bid"]) if "visible_levels_bid" in history else 0.0
    visible_levels_ask = safe_mean(history["visible_levels_ask"]) if "visible_levels_ask" in history else 0.0
    one_sided_ratio = one_sided_duration_ratio(history)
    drought_age = safe_mean(history["drought_age"]) if "drought_age" in history else 0.0
    recovery_pressure = safe_mean(history["recovery_pressure"]) if "recovery_pressure" in history else 0.0
    impact_residue = safe_mean(history["impact_residue"]) if "impact_residue" in history else 0.0
    regime_dwell = safe_mean(history["regime_dwell"]) if "regime_dwell" in history else 0.0
    depletion_recovery_half_life = recovery_half_life(history["recovery_pressure"]) if "recovery_pressure" in history else 0.0
    meta_active_impact, meta_decay_ratio = meta_impact_decay_metrics(history, debug)
    shock_decay_ratio = shock_decay_metrics(history, debug)
    directional_regime_share = float((history["regime"] == "directional").mean()) if not history.empty else 0.0
    stressed_regime_share = float((history["regime"] == "stressed").mean()) if not history.empty else 0.0
    quote_revision_burstiness, revision_spread_ratio, refill_recovery_ratio = revision_wave_metrics(history, debug)
    maker_stress_mean = safe_mean(debug["maker_stress"]) if "maker_stress" in debug else 0.0
    flow_toxicity_mean = safe_mean(debug["flow_toxicity"]) if "flow_toxicity" in debug else 0.0
    refill_pressure_mean = safe_mean(debug["refill_pressure"]) if "refill_pressure" in debug else 0.0

    return {
        "stage": stage,
        "preset": preset,
        "seed": int(seed),
        "config_label": config_label,
        "knob_name": knob_name,
        "knob_scale": float(knob_scale) if knob_scale is not None else None,
        "repeat_idx": int(repeat_idx) if repeat_idx is not None else None,
        "steps": int(run.requested_steps),
        "warmup_fraction": float(run.warmup_fraction),
        "warmup_cutoff_step": int(run.warmup_cutoff_step),
        "analysis_steps": int(effective_steps),
        "mean_return": safe_mean(mid_return),
        "return_std": safe_std(mid_return),
        "realized_vol": float(realized_vol),
        "abs_return_acf1": safe_autocorr(abs_return, lag=1),
        "abs_return_acf5": safe_autocorr(abs_return, lag=5),
        "mean_spread": safe_mean(history["spread"]) if not history.empty else 0.0,
        "median_spread": float(history["spread"].median()) if not history.empty else 0.0,
        "spread_unique_count": int(history["spread"].nunique()) if not history.empty else 0,
        "spread_gt_min_tick_ratio": float((history["spread"] > tick_size_from_run(run)).mean()) if not history.empty else 0.0,
        "crossing_violation_count": int(crossing_history + crossing_events),
        "mean_depth_imbalance": safe_mean(history["depth_imbalance"]) if not history.empty else 0.0,
        "depth_imbalance_std": safe_std(history["depth_imbalance"]) if not history.empty else 0.0,
        "imbalance_next_mid_return_corr": safe_corr(history["depth_imbalance"], next_mid_return) if not history.empty else 0.0,
        "events_per_step": float(event_count / effective_steps),
        "events_per_step_market": float(market_event_count / effective_steps),
        "market_buy_share": float(buy_count / max(market_event_count, 1)),
        "market_sell_share": float(sell_count / max(market_event_count, 1)),
        "cancel_share": float(len(cancel_events) / max(event_count, 1)),
        "limit_share": float(len(limit_events) / max(event_count, 1)),
        "buy_event_count_acf1": safe_autocorr(
            per_step_event_count(step_index, market_events.loc[market_events["side"] == "buy"]),
            lag=1,
        ),
        "cancel_event_count_acf1": safe_autocorr(per_step_event_count(step_index, cancel_events), lag=1),
        "trade_sign_acf1": safe_autocorr(market_signs, lag=1),
        "directional_ratio": directional_ratio(market_signs),
        "phase_open_mean_spread": phase_metrics["phase_open_mean_spread"],
        "phase_mid_mean_spread": phase_metrics["phase_mid_mean_spread"],
        "phase_close_mean_spread": phase_metrics["phase_close_mean_spread"],
        "phase_open_market_order_intensity": phase_metrics["phase_open_market_order_intensity"],
        "phase_mid_market_order_intensity": phase_metrics["phase_mid_market_order_intensity"],
        "phase_close_market_order_intensity": phase_metrics["phase_close_market_order_intensity"],
        "phase_open_total_fill_qty": phase_metrics["phase_open_total_fill_qty"],
        "phase_mid_total_fill_qty": phase_metrics["phase_mid_total_fill_qty"],
        "phase_close_total_fill_qty": phase_metrics["phase_close_total_fill_qty"],
        "phase_open_realized_vol": phase_metrics["phase_open_realized_vol"],
        "phase_mid_realized_vol": phase_metrics["phase_mid_realized_vol"],
        "phase_close_realized_vol": phase_metrics["phase_close_realized_vol"],
        "phase_spread_range": phase_metrics["phase_spread_range"],
        "phase_fill_range": phase_metrics["phase_fill_range"],
        "open_close_activity_ratio": phase_metrics["open_close_activity_ratio"],
        "shock_abs_return": float(shock_abs_return),
        "calm_abs_return": float(calm_abs_return),
        "shock_to_calm_ratio": float(shock_abs_return / max(calm_abs_return, EPSILON)),
        "shock_impact_ratio": float(shock_abs_return / max(calm_abs_return, EPSILON)),
        "shock_decay_ratio": float(shock_decay_ratio),
        "meta_active_directional_ratio": float(meta_active_ratio),
        "meta_inactive_directional_ratio": float(meta_inactive_ratio),
        "meta_active_impact_ratio": float(meta_active_impact),
        "meta_impact_ratio": float(meta_active_impact),
        "meta_decay_ratio": float(meta_decay_ratio),
        "regime_switch_rate": regime_switch_rate(history),
        "directional_regime_share": directional_regime_share,
        "stressed_regime_share": stressed_regime_share,
        "spread_q90": float(spread_q90),
        "spread_q99": float(spread_q99),
        "abs_return_q90": float(abs_return_q90),
        "abs_return_q99": float(abs_return_q99),
        "return_tail_ratio": float(abs_return_q99 / max(abs_return_q90, EPSILON)),
        "visible_levels_bid": float(visible_levels_bid),
        "visible_levels_ask": float(visible_levels_ask),
        "one_sided_ratio": float(one_sided_ratio),
        "one_sided_step_ratio": float(one_sided_ratio),
        "drought_age": float(drought_age),
        "recovery_pressure": float(recovery_pressure),
        "impact_residue": float(impact_residue),
        "regime_dwell": float(regime_dwell),
        "depletion_recovery_half_life": float(depletion_recovery_half_life),
        "maker_stress_mean": float(maker_stress_mean),
        "flow_toxicity_mean": float(flow_toxicity_mean),
        "refill_pressure_mean": float(refill_pressure_mean),
        "quote_revision_burstiness": float(quote_revision_burstiness),
        "revision_spread_ratio": float(revision_spread_ratio),
        "refill_recovery_ratio": float(refill_recovery_ratio),
        "steps_per_second": float(run.requested_steps / max(run.elapsed_seconds, EPSILON)),
        "events_logged_per_second": float(logged_event_count / max(run.elapsed_seconds, EPSILON)),
        "peak_memory_mb": float(run.peak_memory_mb),
        "bytes_per_logged_event": float((run.peak_memory_mb * 1024.0 * 1024.0) / max(logged_event_count, 1)),
        "memory_metric": run.memory_metric,
        "memory_growth_without_recovery": bool(run.memory_growth_without_recovery),
        "run_failed": bool(run.run_failed),
        "error_message": run.error_message or "",
    }


def analysis_history(run: ValidationRun) -> pd.DataFrame:
    if run.history.empty:
        return run.history
    start_step = analysis_start_step(run)
    history = run.history.loc[run.history["step"] >= start_step].copy()
    return history if not history.empty else run.history.copy()


def analysis_events(run: ValidationRun) -> pd.DataFrame:
    if run.event_history.empty:
        return run.event_history
    start_step = analysis_start_step(run)
    events = run.event_history.loc[run.event_history["step"] >= start_step].copy()
    return events if not events.empty else run.event_history.copy()


def analysis_debug(run: ValidationRun) -> pd.DataFrame:
    if run.debug_history.empty:
        return run.debug_history
    start_step = analysis_start_step(run)
    debug = run.debug_history.loc[run.debug_history["step"] >= start_step].copy()
    return debug if not debug.empty else run.debug_history.copy()


def analysis_start_step(run: ValidationRun) -> int:
    if run.requested_steps <= 0:
        return 0
    return int(min(max(run.warmup_cutoff_step, 1), run.requested_steps))


def analysis_step_index(run: ValidationRun) -> np.ndarray:
    if run.requested_steps <= 0:
        return np.array([0], dtype=int)
    start_step = analysis_start_step(run)
    if start_step > run.requested_steps:
        start_step = run.requested_steps
    return np.arange(start_step, run.requested_steps + 1, dtype=int)


def phase_metrics_by_session(history: pd.DataFrame, events: pd.DataFrame, market_events: pd.DataFrame) -> dict[str, float]:
    rows: dict[str, float] = {}
    spreads: list[float] = []
    fills: list[float] = []
    for phase in PHASE_ORDER:
        phase_history = history.loc[history["session_phase"] == phase]
        phase_events = events.loc[events["session_phase"] == phase]
        phase_market_events = market_events.loc[market_events["session_phase"] == phase]
        phase_steps = max(len(phase_history), 1)
        mean_spread = safe_mean(phase_history["spread"]) if not phase_history.empty else 0.0
        total_fill_qty = float(phase_events["fill_qty"].sum()) if not phase_events.empty else 0.0
        rows[f"phase_{phase}_mean_spread"] = mean_spread
        rows[f"phase_{phase}_market_order_intensity"] = float(len(phase_market_events) / phase_steps)
        rows[f"phase_{phase}_total_fill_qty"] = total_fill_qty
        rows[f"phase_{phase}_realized_vol"] = safe_mean(phase_history["realized_vol"]) if not phase_history.empty else 0.0
        spreads.append(mean_spread)
        fills.append(total_fill_qty)
    rows["phase_spread_range"] = float(max(spreads) - min(spreads)) if spreads else 0.0
    rows["phase_fill_range"] = float(max(fills) - min(fills)) if fills else 0.0
    mid_fill = rows["phase_mid_total_fill_qty"]
    rows["open_close_activity_ratio"] = float(
        (rows["phase_open_total_fill_qty"] + rows["phase_close_total_fill_qty"]) / max(2.0 * mid_fill, EPSILON)
    )
    return rows


def shock_vs_calm_abs_return(history: pd.DataFrame, debug: pd.DataFrame) -> tuple[float, float]:
    if history.empty:
        return 0.0, 0.0
    if debug.empty:
        calm = history["mid_price"].diff().abs().fillna(0.0)
        return 0.0, safe_mean(calm)

    per_step_shock = (
        debug.groupby("step")["shock_state"]
        .agg(lambda states: "none" if (states == "none").all() else next(value for value in states if value != "none"))
        .rename("shock_state")
    )
    joined = history.set_index("step").join(per_step_shock, how="left").fillna({"shock_state": "none"})
    abs_return = joined["mid_price"].diff().abs().fillna(0.0)
    shock_abs = abs_return.loc[joined["shock_state"] != "none"]
    calm_abs = abs_return.loc[joined["shock_state"] == "none"]
    return safe_mean(shock_abs), safe_mean(calm_abs)


def meta_directional_ratios(joined: pd.DataFrame) -> tuple[float, float]:
    if joined.empty:
        return 0.0, 0.0
    market_joined = joined.loc[joined["event_type"] == "market"].copy()
    if market_joined.empty:
        return 0.0, 0.0
    signs = market_joined["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)
    active = signs.loc[market_joined["meta_order_id"].notna()]
    inactive = signs.loc[market_joined["meta_order_id"].isna()]
    return directional_ratio(active), directional_ratio(inactive)


def per_step_event_count(step_index: np.ndarray, events: pd.DataFrame) -> pd.Series:
    if len(step_index) == 0:
        return pd.Series(dtype=float)
    counts = events.groupby("step")["event_type"].count() if not events.empty else pd.Series(dtype=float)
    return counts.reindex(step_index, fill_value=0.0).astype(float)


def join_events_and_debug(events: pd.DataFrame, debug: pd.DataFrame) -> pd.DataFrame:
    if events.empty or debug.empty:
        return pd.DataFrame()
    return events.merge(debug, on=["step", "event_idx"], how="inner", suffixes=("_event", "_debug"))


def regime_switch_rate(history: pd.DataFrame) -> float:
    if len(history) < 2:
        return 0.0
    switches = history["regime"].ne(history["regime"].shift()).iloc[1:]
    return float(switches.mean())


def _series_quantile(values: pd.Series, q: float) -> float:
    clean = values.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(q))


def one_sided_duration_ratio(history: pd.DataFrame) -> float:
    if history.empty or "visible_levels_bid" not in history or "visible_levels_ask" not in history:
        return 0.0
    return float(((history["visible_levels_bid"] <= 1) | (history["visible_levels_ask"] <= 1)).mean())


def recovery_half_life(recovery_pressure: pd.Series) -> float:
    clean = recovery_pressure.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)
    if len(clean) < 4:
        return 0.0
    half_lives: list[int] = []
    values = clean.to_numpy(dtype=float)
    for index in range(1, len(values) - 1):
        peak = values[index]
        if peak < 0.25 or peak < values[index - 1] or peak < values[index + 1]:
            continue
        threshold = peak * 0.5
        for future_index in range(index + 1, len(values)):
            if values[future_index] <= threshold:
                half_lives.append(future_index - index)
                break
    return safe_mean(half_lives)


def meta_impact_decay_metrics(history: pd.DataFrame, debug: pd.DataFrame) -> tuple[float, float]:
    if history.empty or debug.empty:
        return 0.0, 0.0
    per_step_meta_active = debug.groupby("step")["meta_order_id"].apply(lambda values: bool(values.notna().any())).rename("meta_active")
    joined = history.set_index("step").join(per_step_meta_active, how="left").fillna({"meta_active": False})
    abs_return = joined["mid_price"].diff().abs().fillna(0.0)
    active = abs_return.loc[joined["meta_active"].astype(bool)]
    inactive = abs_return.loc[~joined["meta_active"].astype(bool)]
    impact_ratio = safe_mean(active) / max(safe_mean(inactive), EPSILON)

    meta_state = joined["meta_active"].astype(bool)
    decay_values: list[float] = []
    for index in range(1, len(joined) - 2):
        if meta_state.iloc[index - 1] and not meta_state.iloc[index]:
            decay_window = abs_return.iloc[index : index + 3]
            decay_values.append(safe_mean(decay_window) / max(safe_mean(active), EPSILON))
    return float(impact_ratio), float(safe_mean(decay_values))


def shock_decay_metrics(history: pd.DataFrame, debug: pd.DataFrame) -> float:
    if history.empty or debug.empty:
        return 0.0
    per_step_shock = debug.groupby("step")["shock_state"].agg(lambda values: any(value != "none" for value in values)).rename("shock_active")
    joined = history.set_index("step").join(per_step_shock, how="left").fillna({"shock_active": False})
    abs_return = joined["mid_price"].diff().abs().fillna(0.0)
    active = abs_return.loc[joined["shock_active"].astype(bool)]
    if active.empty:
        return 0.0
    decay_values: list[float] = []
    shock_state = joined["shock_active"].astype(bool)
    for index in range(1, len(joined) - 2):
        if shock_state.iloc[index - 1] and not shock_state.iloc[index]:
            decay_window = abs_return.iloc[index : index + 3]
            decay_values.append(safe_mean(decay_window) / max(safe_mean(active), EPSILON))
    return float(safe_mean(decay_values))


def revision_wave_metrics(history: pd.DataFrame, debug: pd.DataFrame) -> tuple[float, float, float]:
    required = {"quote_revision_wave", "refill_pressure"}
    if history.empty or debug.empty or not required <= set(debug.columns):
        return 0.0, 0.0, 0.0

    per_step = debug.groupby("step").agg(
        quote_revision_wave=("quote_revision_wave", "max"),
        refill_pressure=("refill_pressure", "mean"),
    )
    joined = history.set_index("step").join(per_step, how="left")
    joined["quote_revision_wave"] = joined["quote_revision_wave"].fillna(False).astype(bool)
    joined["refill_pressure"] = joined["refill_pressure"].ffill().bfill().fillna(0.0)
    revision_steps = joined["quote_revision_wave"].astype(float)
    burstiness = safe_autocorr(revision_steps, lag=1)

    revision_spread = safe_mean(joined.loc[joined["quote_revision_wave"], "spread"])
    calm_spread = safe_mean(joined.loc[~joined["quote_revision_wave"], "spread"])
    spread_ratio = float(revision_spread / max(calm_spread, EPSILON))

    total_depth = joined["top_n_bid_qty"].astype(float) + joined["top_n_ask_qty"].astype(float)
    refill_ratios: list[float] = []
    wave_flags = joined["quote_revision_wave"].to_numpy(dtype=bool)
    depth_values = total_depth.to_numpy(dtype=float)
    for index, active in enumerate(wave_flags):
        if not active:
            continue
        future_index = min(index + 3, len(depth_values) - 1)
        refill_ratios.append(depth_values[future_index] / max(depth_values[index], EPSILON))
    return float(burstiness), float(spread_ratio), float(safe_mean(refill_ratios))


__all__ = [
    "analysis_debug",
    "analysis_events",
    "analysis_history",
    "analysis_start_step",
    "analysis_step_index",
    "compute_run_metrics",
    "join_events_and_debug",
    "meta_directional_ratios",
    "per_step_event_count",
    "phase_metrics_by_session",
    "regime_switch_rate",
    "revision_wave_metrics",
    "shock_vs_calm_abs_return",
]
