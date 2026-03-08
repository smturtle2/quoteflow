from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .shared import (
    EPSILON,
    INVARIANT_FAILURE_COLUMNS,
    OPTIONAL_NUMERIC_COLUMNS,
    PHASE_ORDER,
    ValidationRun,
    coefficient_of_variation,
    deterministic_metric_view,
    directional_ratio,
    failure_row,
    quasi_monotonic,
    safe_autocorr,
    safe_corr,
    safe_mean,
    safe_std,
    sensitivity_target_metric,
    stable_frame_hash,
    stable_object_hash,
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
    event_count = len(events)
    effective_steps = max(len(step_index), 1)
    logged_event_count = len(run.event_history) + len(run.debug_history)

    mid_return = history["mid_price"].diff().fillna(0.0) if not history.empty else pd.Series(dtype=float)
    next_mid_return = mid_return.shift(-1).fillna(0.0) if not mid_return.empty else pd.Series(dtype=float)
    abs_return = mid_return.abs() if not mid_return.empty else pd.Series(dtype=float)

    market_events = events.loc[events["event_type"] == "market"].copy()
    limit_events = events.loc[events["event_type"] == "limit"].copy()
    cancel_events = events.loc[events["event_type"] == "cancel"].copy()

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
    crossing_events = int((run.event_history["best_bid_after"] >= run.event_history["best_ask_after"]).sum()) if not run.event_history.empty else 0
    realized_vol = safe_mean(history["realized_vol"]) if not history.empty else 0.0
    if realized_vol <= 0.0:
        realized_vol = safe_std(mid_return)

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
        "buy_event_count_acf1": safe_autocorr(per_step_event_count(step_index, market_events.loc[market_events["side"] == "buy"]), lag=1),
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
        "shock_abs_return": float(shock_abs_return),
        "calm_abs_return": float(calm_abs_return),
        "shock_to_calm_ratio": float(shock_abs_return / max(calm_abs_return, EPSILON)),
        "meta_active_directional_ratio": float(meta_active_ratio),
        "meta_inactive_directional_ratio": float(meta_inactive_ratio),
        "regime_switch_rate": regime_switch_rate(history),
        "steps_per_second": float(run.requested_steps / max(run.elapsed_seconds, EPSILON)),
        "events_logged_per_second": float(logged_event_count / max(run.elapsed_seconds, EPSILON)),
        "peak_memory_mb": float(run.peak_memory_mb),
        "bytes_per_logged_event": float((run.peak_memory_mb * 1024.0 * 1024.0) / max(logged_event_count, 1)),
        "memory_metric": run.memory_metric,
        "memory_growth_without_recovery": bool(run.memory_growth_without_recovery),
        "run_failed": bool(run.run_failed),
        "error_message": run.error_message or "",
    }


def summarize_validation_grid(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate preset-level summary statistics across seeds."""

    if run_metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    summary_metrics = (
        "mean_spread",
        "median_spread",
        "realized_vol",
        "abs_return_acf1",
        "abs_return_acf5",
        "imbalance_next_mid_return_corr",
        "events_per_step",
        "events_per_step_market",
        "market_buy_share",
        "market_sell_share",
        "cancel_share",
        "limit_share",
        "buy_event_count_acf1",
        "cancel_event_count_acf1",
        "trade_sign_acf1",
        "directional_ratio",
        "phase_spread_range",
        "phase_fill_range",
        "shock_to_calm_ratio",
        "meta_active_directional_ratio",
        "meta_inactive_directional_ratio",
        "regime_switch_rate",
        "steps_per_second",
        "peak_memory_mb",
        "bytes_per_logged_event",
        "spread_unique_count",
    )
    cv_metrics = (
        "mean_spread",
        "realized_vol",
        "events_per_step",
        "imbalance_next_mid_return_corr",
        "abs_return_acf1",
    )

    for (stage, preset), frame in run_metrics.groupby(["stage", "preset"], sort=False):
        row: dict[str, Any] = {
            "stage": stage,
            "preset": preset,
            "runs": int(len(frame)),
            "run_failures": int(frame["run_failed"].sum()),
            "memory_growth_failures": int(frame["memory_growth_without_recovery"].sum()),
        }
        for metric in summary_metrics:
            if metric in frame.columns:
                row[f"{metric}_mean"] = safe_mean(frame[metric])
                row[f"{metric}_std"] = safe_std(frame[metric])
            else:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_std"] = 0.0
        for metric in cv_metrics:
            row[f"{metric}_cv"] = coefficient_of_variation(row[f"{metric}_mean"], row[f"{metric}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sensitivity_grid(sensitivity_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sensitivity runs by knob and scale."""

    if sensitivity_metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (knob_name, knob_scale), frame in sensitivity_metrics.groupby(["knob_name", "knob_scale"], sort=False):
        row: dict[str, Any] = {
            "knob_name": knob_name,
            "knob_scale": float(knob_scale),
            "runs": int(len(frame)),
            "limit_share_mean": safe_mean(frame["limit_share"]),
            "events_per_step_market_mean": safe_mean(frame["events_per_step_market"]),
            "cancel_share_mean": safe_mean(frame["cancel_share"]),
            "realized_vol_mean": safe_mean(frame["realized_vol"]),
            "regime_switch_rate_mean": safe_mean(frame["regime_switch_rate"]),
            "phase_structure_signal_mean": safe_mean(frame["phase_spread_range"] + frame["phase_fill_range"]),
            "event_clustering_mean": safe_mean((frame["buy_event_count_acf1"] + frame["cancel_event_count_acf1"]) / 2.0),
            "meta_active_directional_ratio_mean": safe_mean(frame["meta_active_directional_ratio"]),
            "shock_to_calm_ratio_mean": safe_mean(frame["shock_to_calm_ratio"]),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    direction_rows: list[dict[str, Any]] = []
    for knob_name, frame in summary.groupby("knob_name", sort=False):
        ordered = frame.sort_values("knob_scale").reset_index(drop=True)
        target_metric = sensitivity_target_metric(knob_name)
        values = ordered[target_metric].to_numpy(dtype=float)
        direction_ok = quasi_monotonic(values)
        if (ordered["knob_scale"] == 1.0).any():
            baseline_value = float(ordered.loc[ordered["knob_scale"] == 1.0, target_metric].iloc[0])
        else:
            baseline_value = float(values[0])
        if (ordered["knob_scale"] == 2.0).any():
            high_value = float(ordered.loc[ordered["knob_scale"] == 2.0, target_metric].iloc[0])
        else:
            high_value = float(values[-1])
        direction_rows.append(
            {
                "knob_name": knob_name,
                "target_metric": target_metric,
                "direction_ok": bool(direction_ok and (high_value > baseline_value)),
            }
        )

    return summary.merge(pd.DataFrame(direction_rows), on="knob_name", how="left")


def collect_invariant_failures(
    run: ValidationRun,
    *,
    stage: str,
    preset: str,
    seed: int,
) -> pd.DataFrame:
    """Collect normalized invariant failures for one run."""

    failures: list[dict[str, Any]] = []

    if run.run_failed:
        failures.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name="run_failed",
                details=run.error_message or "run failed",
            )
        )

    if not event_order_ok(run.event_history):
        failures.extend(
            event_order_failures(
                run.event_history,
                preset=preset,
                seed=seed,
                stage=stage,
            )
        )

    if not run.event_history.empty:
        crossing = run.event_history.loc[run.event_history["best_bid_after"] >= run.event_history["best_ask_after"]]
        for row in crossing.itertuples(index=False):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=row.event_idx,
                    invariant_name="crossed_or_locked_book_event",
                    details=f"best_bid_after={row.best_bid_after} best_ask_after={row.best_ask_after}",
                )
            )

        market_events = run.event_history.loc[run.event_history["event_type"] == "market"]
        fill_sums = market_events["fills"].apply(lambda fills: float(sum(qty for _, qty in fills)))
        mismatched = market_events.loc[
            ~np.isclose(
                market_events["fill_qty"].to_numpy(dtype=float),
                fill_sums.to_numpy(dtype=float),
            )
        ]
        for row in mismatched.itertuples(index=False):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=row.event_idx,
                    invariant_name="market_fill_mismatch",
                    details=f"fill_qty={row.fill_qty}",
                )
            )

    if not run.history.empty:
        crossing = run.history.loc[run.history["best_bid"] >= run.history["best_ask"]]
        for row in crossing.itertuples(index=False):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=np.nan,
                    invariant_name="crossed_or_locked_book_history",
                    details=f"best_bid={row.best_bid} best_ask={row.best_ask}",
                )
            )

    failures.extend(
        debug_alignment_failures(
            run.debug_history,
            run.event_history,
            preset=preset,
            seed=seed,
            stage=stage,
        )
    )

    for label, snapshot in (("start_snapshot", run.start_snapshot), ("end_snapshot", run.end_snapshot)):
        failures.extend(snapshot_failures(snapshot, preset=preset, seed=seed, stage=stage, label=label))

    visual_rows = getattr(run.market, "_visual_history", []) if run.market is not None else []
    for index, row in enumerate(visual_rows):
        bid_values = np.asarray(row.bid_qty, dtype=float)
        ask_values = np.asarray(row.ask_qty, dtype=float)
        if np.any(bid_values[np.isfinite(bid_values)] < 0.0):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=np.nan,
                    invariant_name="negative_visual_bid_depth",
                    details=f"visual_row={index}",
                )
            )
        if np.any(ask_values[np.isfinite(ask_values)] < 0.0):
            failures.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=np.nan,
                    invariant_name="negative_visual_ask_depth",
                    details=f"visual_row={index}",
                )
            )

    return pd.DataFrame.from_records(failures, columns=INVARIANT_FAILURE_COLUMNS)


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


def event_order_ok(events: pd.DataFrame) -> bool:
    if len(events) < 2:
        return True
    steps = events["step"].to_numpy(dtype=int)
    event_idx = events["event_idx"].to_numpy(dtype=int)
    return bool(np.all((steps[1:] > steps[:-1]) | ((steps[1:] == steps[:-1]) & (event_idx[1:] > event_idx[:-1]))))


def event_order_failures(
    events: pd.DataFrame,
    *,
    preset: str,
    seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(events) < 2:
        return rows
    steps = events["step"].to_numpy(dtype=int)
    event_idx = events["event_idx"].to_numpy(dtype=int)
    invalid = np.where(~((steps[1:] > steps[:-1]) | ((steps[1:] == steps[:-1]) & (event_idx[1:] > event_idx[:-1]))))[0]
    for index in invalid:
        row = events.iloc[index + 1]
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=row["step"],
                event_idx=row["event_idx"],
                invariant_name="event_order_nonmonotonic",
                details="(step, event_idx) ordering broken",
            )
        )
    return rows


def debug_alignment_failures(
    debug: pd.DataFrame,
    events: pd.DataFrame,
    *,
    preset: str,
    seed: int,
    stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(debug) != len(events):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name="debug_length_mismatch",
                details=f"events={len(events)} debug={len(debug)}",
            )
        )
        return rows
    if debug.empty and events.empty:
        return rows
    joined = events.merge(debug, on=["step", "event_idx"], how="outer", indicator=True, suffixes=("_event", "_debug"))
    mismatched = joined.loc[joined["_merge"] != "both"]
    for row in mismatched.itertuples(index=False):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=row.step,
                event_idx=row.event_idx,
                invariant_name="debug_event_join_mismatch",
                details=str(row._merge),
            )
        )
    both = joined.loc[joined["_merge"] == "both"]
    if not both.empty:
        phase_mismatch = both.loc[both["session_phase_event"] != both["session_phase_debug"]]
        for row in phase_mismatch.itertuples(index=False):
            rows.append(
                failure_row(
                    preset=preset,
                    seed=seed,
                    stage=stage,
                    step=row.step,
                    event_idx=row.event_idx,
                    invariant_name="debug_phase_mismatch",
                    details=f"event={row.session_phase_event} debug={row.session_phase_debug}",
                )
            )
    return rows


def snapshot_failures(
    snapshot: dict[str, Any] | None,
    *,
    preset: str,
    seed: int,
    stage: str,
    label: str,
) -> list[dict[str, Any]]:
    if snapshot is None:
        return [
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name=f"{label}_missing",
                details="snapshot missing",
            )
        ]

    rows: list[dict[str, Any]] = []
    bids = snapshot.get("bids", [])
    asks = snapshot.get("asks", [])
    bid_prices = [level["price"] for level in bids]
    ask_prices = [level["price"] for level in asks]
    bid_qty = [level["qty"] for level in bids]
    ask_qty = [level["qty"] for level in asks]
    step = snapshot.get("step", np.nan)

    if any(quantity < 0.0 for quantity in bid_qty):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_negative_bid_depth",
                details="negative bid depth",
            )
        )
    if any(quantity < 0.0 for quantity in ask_qty):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_negative_ask_depth",
                details="negative ask depth",
            )
        )
    if bid_prices != sorted(bid_prices, reverse=True):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_bid_sorting",
                details="bid prices not descending",
            )
        )
    if ask_prices != sorted(ask_prices):
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_ask_sorting",
                details="ask prices not ascending",
            )
        )
    if bids and asks and bid_prices[0] >= ask_prices[0]:
        rows.append(
            failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=step,
                event_idx=np.nan,
                invariant_name=f"{label}_crossed_book",
                details=f"best_bid={bid_prices[0]} best_ask={ask_prices[0]}",
            )
        )
    return rows


def table_nonfinite_failures(frame: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    failures: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    numeric = frame.select_dtypes(include=[np.number])
    if stage == "run_metrics":
        numeric = numeric.drop(columns=list(OPTIONAL_NUMERIC_COLUMNS & set(numeric.columns)), errors="ignore")
    if numeric.empty:
        return pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
    invalid_rows, invalid_cols = np.where(invalid)
    for row_index, col_index in zip(invalid_rows, invalid_cols):
        column = numeric.columns[col_index]
        seed_value = -1
        if "seed" in frame.columns and pd.notna(frame.iloc[row_index].get("seed")):
            seed_value = int(frame.iloc[row_index]["seed"])
        failures.append(
            failure_row(
                preset=str(frame.iloc[row_index].get("preset", "n/a")),
                seed=seed_value,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name="summary_nonfinite",
                details=f"{column} is non-finite",
            )
        )
    return pd.DataFrame.from_records(failures, columns=INVARIANT_FAILURE_COLUMNS)


def reproducibility_failures(reproducibility: pd.DataFrame) -> pd.DataFrame:
    failures: list[dict[str, Any]] = []
    for row in reproducibility.itertuples(index=False):
        if row.all_reproducible:
            continue
        for field_name in (
            "history_hash_equal",
            "event_hash_equal",
            "debug_hash_equal",
            "metrics_hash_equal",
            "gen_vs_step_history_equal",
            "gen_vs_step_event_equal",
            "gen_vs_step_debug_equal",
        ):
            if getattr(row, field_name):
                continue
            failures.append(
                failure_row(
                    preset=row.preset,
                    seed=row.seed,
                    stage="reproducibility",
                    step=np.nan,
                    event_idx=np.nan,
                    invariant_name=field_name,
                    details="reproducibility mismatch",
                )
            )
    return pd.DataFrame.from_records(failures, columns=INVARIANT_FAILURE_COLUMNS)


def regime_switch_rate(history: pd.DataFrame) -> float:
    if len(history) < 2:
        return 0.0
    switches = history["regime"].ne(history["regime"].shift()).iloc[1:]
    return float(switches.mean())
