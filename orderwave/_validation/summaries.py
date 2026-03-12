from __future__ import annotations

from typing import Any

import pandas as pd

from .shared import (
    coefficient_of_variation,
    quasi_monotonic,
    safe_mean,
    safe_std,
    sensitivity_target_metric,
)


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
        "open_close_activity_ratio",
        "shock_to_calm_ratio",
        "shock_impact_ratio",
        "shock_decay_ratio",
        "meta_active_directional_ratio",
        "meta_inactive_directional_ratio",
        "meta_active_impact_ratio",
        "meta_impact_ratio",
        "meta_decay_ratio",
        "regime_switch_rate",
        "directional_regime_share",
        "stressed_regime_share",
        "spread_q90",
        "spread_q99",
        "abs_return_q90",
        "abs_return_q99",
        "return_tail_ratio",
        "visible_levels_bid",
        "visible_levels_ask",
        "one_sided_ratio",
        "one_sided_step_ratio",
        "drought_age",
        "recovery_pressure",
        "impact_residue",
        "regime_dwell",
        "depletion_recovery_half_life",
        "maker_stress_mean",
        "flow_toxicity_mean",
        "refill_pressure_mean",
        "quote_revision_burstiness",
        "revision_spread_ratio",
        "refill_recovery_ratio",
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
            "limit_share_mean": safe_mean(_column_or_zero(frame, "limit_share")),
            "events_per_step_market_mean": safe_mean(_column_or_zero(frame, "events_per_step_market")),
            "cancel_share_mean": safe_mean(_column_or_zero(frame, "cancel_share")),
            "realized_vol_mean": safe_mean(_column_or_zero(frame, "realized_vol")),
            "regime_switch_rate_mean": safe_mean(_column_or_zero(frame, "regime_switch_rate")),
            "phase_structure_signal_mean": safe_mean(_column_or_zero(frame, "phase_spread_range") + _column_or_zero(frame, "phase_fill_range")),
            "event_clustering_mean": safe_mean((_column_or_zero(frame, "buy_event_count_acf1") + _column_or_zero(frame, "cancel_event_count_acf1")) / 2.0),
            "meta_active_directional_ratio_mean": safe_mean(_column_or_zero(frame, "meta_active_directional_ratio")),
            "meta_active_impact_ratio_mean": safe_mean(_column_with_fallback(frame, "meta_active_impact_ratio", "meta_active_directional_ratio")),
            "meta_impact_ratio_mean": safe_mean(_column_with_fallback(frame, "meta_impact_ratio", "meta_active_directional_ratio")),
            "shock_to_calm_ratio_mean": safe_mean(_column_or_zero(frame, "shock_to_calm_ratio")),
            "shock_impact_ratio_mean": safe_mean(_column_with_fallback(frame, "shock_impact_ratio", "shock_to_calm_ratio")),
            "return_tail_ratio_mean": safe_mean(_column_or_zero(frame, "return_tail_ratio")),
            "one_sided_ratio_mean": safe_mean(_column_or_zero(frame, "one_sided_ratio")),
            "one_sided_step_ratio_mean": safe_mean(_column_or_zero(frame, "one_sided_step_ratio")),
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


__all__ = ["summarize_sensitivity_grid", "summarize_validation_grid"]


def _column_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(float)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _column_with_fallback(frame: pd.DataFrame, column: str, fallback: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(float)
    return _column_or_zero(frame, fallback)
