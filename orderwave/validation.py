from __future__ import annotations

"""Validation helpers for longer-form orderwave experiment sweeps."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from orderwave.market import Market


PHASE_ORDER: tuple[str, ...] = ("open", "mid", "close")
DEFAULT_SENSITIVITY_KNOBS: tuple[str, ...] = (
    "shock_scale",
    "meta_order_scale",
    "excitation_scale",
    "fair_price_vol_scale",
)
EPSILON = 1e-9


@dataclass(frozen=True)
class ValidationRun:
    """Container for a single simulated run and its output tables."""

    market: Market
    history: pd.DataFrame
    event_history: pd.DataFrame
    debug_history: pd.DataFrame
    elapsed_seconds: float


def run_market_validation(
    *,
    preset: str,
    seed: int,
    steps: int,
    config_overrides: Mapping[str, object] | None = None,
) -> ValidationRun:
    """Run one market simulation and capture its outputs."""

    config: dict[str, object] = {"preset": preset}
    if config_overrides:
        config.update(config_overrides)

    market = Market(seed=seed, config=config)
    started = perf_counter()
    market.gen(steps=steps)
    elapsed_seconds = perf_counter() - started
    return ValidationRun(
        market=market,
        history=market.get_history(),
        event_history=market.get_event_history(),
        debug_history=market.get_debug_history(),
        elapsed_seconds=elapsed_seconds,
    )


def compute_run_metrics(
    run: ValidationRun,
    *,
    preset: str,
    seed: int,
    steps: int,
    config_label: str = "baseline",
    knob_name: str | None = None,
    knob_scale: float | None = None,
) -> dict[str, Any]:
    """Compute validation metrics for one completed run."""

    history = run.history
    events = run.event_history
    debug = run.debug_history

    mid_return = history["mid_price"].diff().fillna(0.0)
    next_mid_return = mid_return.shift(-1).fillna(0.0)
    abs_return = mid_return.abs()
    nonzero_abs_return = abs_return.loc[abs_return > 0.0]
    realized_vol = _safe_mean(history["realized_vol"])
    if realized_vol <= 0.0:
        realized_vol = float(abs_return.mean())

    market_events = events.loc[events["event_type"] == "market"].copy()
    cancel_events = events.loc[events["event_type"] == "cancel"].copy()
    buy_count = int((market_events["side"] == "buy").sum())
    sell_count = int((market_events["side"] == "sell").sum())
    market_event_count = buy_count + sell_count
    market_buy_share = buy_count / market_event_count if market_event_count > 0 else np.nan
    market_sell_share = sell_count / market_event_count if market_event_count > 0 else np.nan

    phase_spread = history.groupby("session_phase")["spread"].mean().reindex(PHASE_ORDER, fill_value=np.nan)
    if events.empty:
        phase_fill = pd.Series(0.0, index=PHASE_ORDER)
    else:
        phase_fill = (
            events.groupby(["session_phase", "step"])["fill_qty"]
            .sum()
            .groupby("session_phase")
            .mean()
            .reindex(PHASE_ORDER, fill_value=0.0)
        )

    market_count_by_step = (
        market_events.groupby("step")["event_type"].count() if not market_events.empty else pd.Series(dtype=float)
    )
    cancel_count_by_step = (
        cancel_events.groupby("step")["event_type"].count() if not cancel_events.empty else pd.Series(dtype=float)
    )

    debug_alignment_ok, joined = _debug_alignment(events, debug)
    if not joined.empty:
        market_joined = joined.loc[joined["event_type"] == "market"].copy()
        market_joined["sign"] = market_joined["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)
        meta_step_sign = market_joined.loc[market_joined["meta_order_id"].notna()].groupby("step")["sign"].sum()
        base_step_sign = market_joined.loc[market_joined["meta_order_id"].isna()].groupby("step")["sign"].sum()
        meta_directionality = _safe_mean(meta_step_sign.abs())
        base_directionality = _safe_mean(base_step_sign.abs())
    else:
        meta_directionality = np.nan
        base_directionality = np.nan

    shock_abs_return, calm_abs_return = _shock_vs_calm_abs_return(history, debug)
    shock_to_calm_ratio = shock_abs_return / max(calm_abs_return, EPSILON) if calm_abs_return > 0.0 else np.nan

    event_order_ok = _event_order_ok(events)
    best_quote_ok = _best_quote_ok(events, history)
    market_fill_match_ok = _market_fill_match_ok(market_events)
    invariants_ok = bool(event_order_ok and best_quote_ok and market_fill_match_ok and debug_alignment_ok)

    signed_flow_bias = float(abs(history["signed_flow"].mean()))
    return {
        "preset": preset,
        "seed": int(seed),
        "steps": int(steps),
        "config_label": config_label,
        "knob_name": knob_name,
        "knob_scale": knob_scale,
        "mean_spread": float(history["spread"].mean()),
        "spread_unique_values": int(history["spread"].nunique()),
        "realized_vol": float(realized_vol),
        "abs_return_acf1": _safe_autocorr(nonzero_abs_return, lag=1),
        "imbalance_next_return_corr": _safe_corr(history["depth_imbalance"], next_mid_return),
        "events_per_step": float(len(events) / max(steps, 1)),
        "market_buy_share": float(market_buy_share),
        "market_sell_share": float(market_sell_share),
        "market_buy_count": int(buy_count),
        "market_sell_count": int(sell_count),
        "market_count_acf1": _safe_autocorr(market_count_by_step, lag=1),
        "cancel_count_acf1": _safe_autocorr(cancel_count_by_step, lag=1),
        "phase_open_spread": float(phase_spread.get("open", np.nan)),
        "phase_mid_spread": float(phase_spread.get("mid", np.nan)),
        "phase_close_spread": float(phase_spread.get("close", np.nan)),
        "phase_open_fill": float(phase_fill.get("open", 0.0)),
        "phase_mid_fill": float(phase_fill.get("mid", 0.0)),
        "phase_close_fill": float(phase_fill.get("close", 0.0)),
        "phase_spread_range": float(phase_spread.max() - phase_spread.min()),
        "phase_fill_range": float(phase_fill.max() - phase_fill.min()),
        "shock_abs_return": float(shock_abs_return),
        "calm_abs_return": float(calm_abs_return),
        "shock_to_calm_abs_return_ratio": float(shock_to_calm_ratio),
        "signed_flow_bias": signed_flow_bias,
        "meta_directionality": float(meta_directionality),
        "base_directionality": float(base_directionality),
        "event_order_ok": event_order_ok,
        "best_quote_ok": best_quote_ok,
        "market_fill_match_ok": market_fill_match_ok,
        "debug_alignment_ok": debug_alignment_ok,
        "invariants_ok": invariants_ok,
        "elapsed_seconds": float(run.elapsed_seconds),
        "steps_per_second": float(steps / max(run.elapsed_seconds, EPSILON)),
    }


def run_validation_grid(
    *,
    presets: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    config_overrides_by_preset: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Run the preset grid and return one row per preset/seed."""

    rows: list[dict[str, Any]] = []
    for preset in presets:
        preset_overrides = (config_overrides_by_preset or {}).get(preset, {})
        for seed in seeds:
            run = run_market_validation(
                preset=preset,
                seed=int(seed),
                steps=steps,
                config_overrides=preset_overrides,
            )
            rows.append(
                compute_run_metrics(
                    run,
                    preset=preset,
                    seed=int(seed),
                    steps=steps,
                )
            )
    return pd.DataFrame(rows)


def summarize_validation_grid(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate preset-level summary statistics across seeds."""

    rows: list[dict[str, Any]] = []
    summary_metrics = (
        "mean_spread",
        "realized_vol",
        "abs_return_acf1",
        "imbalance_next_return_corr",
        "events_per_step",
        "market_buy_share",
        "market_sell_share",
        "market_count_acf1",
        "cancel_count_acf1",
        "phase_spread_range",
        "phase_fill_range",
        "shock_to_calm_abs_return_ratio",
        "signed_flow_bias",
        "meta_directionality",
        "steps_per_second",
    )
    phase_metrics = (
        "phase_open_spread",
        "phase_mid_spread",
        "phase_close_spread",
        "phase_open_fill",
        "phase_mid_fill",
        "phase_close_fill",
    )

    for preset, frame in run_metrics.groupby("preset", sort=False):
        row: dict[str, Any] = {
            "preset": preset,
            "runs": int(len(frame)),
            "invariant_failures": int((~frame["invariants_ok"]).sum()),
            "event_order_failures": int((~frame["event_order_ok"]).sum()),
            "best_quote_failures": int((~frame["best_quote_ok"]).sum()),
            "market_fill_failures": int((~frame["market_fill_match_ok"]).sum()),
            "debug_alignment_failures": int((~frame["debug_alignment_ok"]).sum()),
        }
        for metric in summary_metrics + phase_metrics:
            row[f"{metric}_mean"] = _safe_mean(frame[metric])
            row[f"{metric}_std"] = _safe_std(frame[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def run_reproducibility_checks(
    *,
    presets: Sequence[str],
    seed: int = 42,
    steps: int = 250,
) -> pd.DataFrame:
    """Check same-seed reproducibility and ``gen`` vs repeated ``step`` consistency."""

    rows: list[dict[str, Any]] = []
    for preset in presets:
        market_a = Market(seed=seed, config={"preset": preset})
        market_b = Market(seed=seed, config={"preset": preset})
        market_step = Market(seed=seed, config={"preset": preset})

        market_a.gen(steps=steps)
        market_b.gen(steps=steps)
        for _ in range(steps):
            market_step.step()

        history_same = market_a.get_history().equals(market_b.get_history())
        events_same = market_a.get_event_history().equals(market_b.get_event_history())
        debug_same = market_a.get_debug_history().equals(market_b.get_debug_history())
        gen_vs_step_history = market_a.get_history().equals(market_step.get_history())
        gen_vs_step_events = market_a.get_event_history().equals(market_step.get_event_history())
        rows.append(
            {
                "preset": preset,
                "seed": int(seed),
                "steps": int(steps),
                "same_seed_history": bool(history_same),
                "same_seed_events": bool(events_same),
                "same_seed_debug": bool(debug_same),
                "gen_vs_step_history": bool(gen_vs_step_history),
                "gen_vs_step_events": bool(gen_vs_step_events),
                "all_reproducible": bool(
                    history_same and events_same and debug_same and gen_vs_step_history and gen_vs_step_events
                ),
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity_grid(
    *,
    preset: str,
    seeds: Sequence[int],
    steps: int,
    knobs: Sequence[str] = DEFAULT_SENSITIVITY_KNOBS,
    scales: Sequence[float] = (1.0, 1.5),
) -> pd.DataFrame:
    """Run knob sensitivity experiments for one preset."""

    rows: list[dict[str, Any]] = []
    for knob_name in knobs:
        for knob_scale in scales:
            for seed in seeds:
                run = run_market_validation(
                    preset=preset,
                    seed=int(seed),
                    steps=steps,
                    config_overrides={knob_name: float(knob_scale)},
                )
                rows.append(
                    compute_run_metrics(
                        run,
                        preset=preset,
                        seed=int(seed),
                        steps=steps,
                        config_label=f"{knob_name}={knob_scale:.2f}",
                        knob_name=knob_name,
                        knob_scale=float(knob_scale),
                    )
                )
    return pd.DataFrame(rows)


def summarize_sensitivity_grid(sensitivity_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sensitivity runs by knob and scale."""

    rows: list[dict[str, Any]] = []
    summary_metrics = (
        "mean_spread",
        "realized_vol",
        "events_per_step",
        "market_count_acf1",
        "cancel_count_acf1",
        "shock_to_calm_abs_return_ratio",
        "meta_directionality",
        "signed_flow_bias",
    )
    for (knob_name, knob_scale), frame in sensitivity_metrics.groupby(["knob_name", "knob_scale"], sort=False):
        row: dict[str, Any] = {
            "knob_name": knob_name,
            "knob_scale": float(knob_scale),
            "runs": int(len(frame)),
        }
        row["event_clustering_mean"] = _safe_mean(frame["market_count_acf1"] + frame["cancel_count_acf1"])
        row["event_clustering_std"] = _safe_std(frame["market_count_acf1"] + frame["cancel_count_acf1"])
        for metric in summary_metrics:
            row[f"{metric}_mean"] = _safe_mean(frame[metric])
            row[f"{metric}_std"] = _safe_std(frame[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_validation_results(
    *,
    run_metrics: pd.DataFrame,
    preset_summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    sensitivity_summary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Produce a coarse pass/hold/fail verdict from the validation outputs."""

    preset_frame = preset_summary.set_index("preset")
    invariants_ok = bool((run_metrics["invariants_ok"]).all())
    reproducible_ok = bool(reproducibility["all_reproducible"].all())

    preset_separation_checks = {
        "vol_spread_gt_balanced": _preset_compare(preset_frame, "volatile", "balanced", "mean_spread_mean"),
        "vol_vol_gt_balanced": _preset_compare(preset_frame, "volatile", "balanced", "realized_vol_mean"),
        "trend_bias_gt_balanced": _preset_compare(preset_frame, "trend", "balanced", "signed_flow_bias_mean"),
        "events_ordered": _ordered_preset_metric(
            preset_frame,
            "events_per_step_mean",
            order=("balanced", "trend", "volatile"),
        ),
    }
    preset_separation_ok = sum(bool(value) for value in preset_separation_checks.values()) >= 3

    stability_ok = True
    for _, row in preset_summary.iterrows():
        spread_cv = _coefficient_of_variation(row["mean_spread_mean"], row["mean_spread_std"])
        vol_cv = _coefficient_of_variation(row["realized_vol_mean"], row["realized_vol_std"])
        if spread_cv > 0.6 or vol_cv > 0.75:
            stability_ok = False
            break

    time_structure_checks = {
        "phase_spread_range": bool((preset_summary["phase_spread_range_mean"] > 0.001).all()),
        "phase_fill_range": bool((preset_summary["phase_fill_range_mean"] > 1.0).all()),
        "market_clustering": bool((preset_summary["market_count_acf1_mean"] > 0.0).all()),
        "cancel_clustering": bool((preset_summary["cancel_count_acf1_mean"] > 0.0).all()),
        "shock_vs_calm": bool((preset_summary["shock_to_calm_abs_return_ratio_mean"] > 1.0).all()),
    }
    time_structure_ok = sum(bool(value) for value in time_structure_checks.values()) >= 4

    sensitivity_checks: dict[str, bool] = {}
    if sensitivity_summary is not None and not sensitivity_summary.empty:
        sensitivity_checks = _evaluate_sensitivity_summary(sensitivity_summary)
    sensitivity_ok = (not sensitivity_checks) or all(sensitivity_checks.values())

    strengths: list[str] = []
    weaknesses: list[str] = []
    if invariants_ok:
        strengths.append("event and book invariants held across all validation runs")
    else:
        weaknesses.append("event or book invariants failed in at least one run")
    if preset_separation_ok:
        strengths.append("preset-level path statistics separate in the intended direction")
    else:
        weaknesses.append("preset differences are not clearly separated across key metrics")
    if time_structure_ok:
        strengths.append("session phase, clustering, and shock effects remain visible")
    else:
        weaknesses.append("time structure is weak in one or more diagnostics")
    if stability_ok:
        strengths.append("seed-to-seed variation stays within a usable range")
    else:
        weaknesses.append("seed variation is large relative to the preset signal")
    if reproducible_ok:
        strengths.append("same-seed and gen-vs-step reproducibility checks pass")
    else:
        weaknesses.append("reproducibility checks failed")
    if sensitivity_checks:
        if sensitivity_ok:
            strengths.append("high-level scaling knobs move the output in the expected direction")
        else:
            weaknesses.append("one or more scaling knobs have weak or reversed sensitivity")

    if not invariants_ok or not reproducible_ok:
        adoption = "NO"
        suitability = "not suitable"
    elif preset_separation_ok and stability_ok and time_structure_ok and sensitivity_ok:
        adoption = "YES"
        suitability = "suitable"
    else:
        adoption = "CONDITIONAL"
        suitability = "conditionally suitable"

    return {
        "synthetic_market_state_generator": suitability,
        "adoption": adoption,
        "preset_separation_ok": preset_separation_ok,
        "seed_stability_ok": stability_ok,
        "time_structure_ok": time_structure_ok,
        "invariants_ok": invariants_ok,
        "reproducible_ok": reproducible_ok,
        "sensitivity_ok": sensitivity_ok,
        "preset_separation_checks": preset_separation_checks,
        "time_structure_checks": time_structure_checks,
        "sensitivity_checks": sensitivity_checks,
        "major_strengths": strengths or ["none"],
        "major_weaknesses": weaknesses or ["none"],
    }


def _event_order_ok(events: pd.DataFrame) -> bool:
    if len(events) < 2:
        return True
    steps = events["step"].to_numpy(dtype=int)
    event_idx = events["event_idx"].to_numpy(dtype=int)
    return bool(
        np.all(
            (steps[1:] > steps[:-1])
            | ((steps[1:] == steps[:-1]) & (event_idx[1:] > event_idx[:-1]))
        )
    )


def _best_quote_ok(events: pd.DataFrame, history: pd.DataFrame) -> bool:
    if not events.empty:
        return bool((events["best_bid_after"] < events["best_ask_after"]).all())
    return bool((history["best_bid"] < history["best_ask"]).all())


def _market_fill_match_ok(market_events: pd.DataFrame) -> bool:
    if market_events.empty:
        return True
    fill_sums = market_events["fills"].apply(lambda fills: float(sum(qty for _, qty in fills)))
    return bool(np.allclose(market_events["fill_qty"].to_numpy(dtype=float), fill_sums.to_numpy(dtype=float)))


def _debug_alignment(events: pd.DataFrame, debug: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    if len(events) != len(debug):
        return False, pd.DataFrame()
    if events.empty and debug.empty:
        return True, pd.DataFrame()
    joined = events.merge(debug, on=["step", "event_idx"], how="inner")
    if len(joined) != len(events):
        return False, joined
    if not joined["session_phase_x"].equals(joined["session_phase_y"]):
        return False, joined
    return True, joined


def _shock_vs_calm_abs_return(history: pd.DataFrame, debug: pd.DataFrame) -> tuple[float, float]:
    if debug.empty:
        mean_abs_return = float(history["mid_price"].diff().abs().fillna(0.0).mean())
        return 0.0, mean_abs_return

    step_shock = debug.groupby("step")["shock_state"].agg(
        lambda states: "none" if (states == "none").all() else next(value for value in states if value != "none")
    )
    step_view = history.set_index("step").join(step_shock.rename("shock_state")).fillna({"shock_state": "none"})
    shock_steps = step_view.loc[step_view["shock_state"] != "none"]
    calm_steps = step_view.loc[step_view["shock_state"] == "none"]
    shock_abs_return = float(shock_steps["mid_price"].diff().abs().mean()) if len(shock_steps) > 1 else 0.0
    calm_abs_return = float(calm_steps["mid_price"].diff().abs().mean()) if len(calm_steps) > 1 else 0.0
    return shock_abs_return, calm_abs_return


def _evaluate_sensitivity_summary(sensitivity_summary: pd.DataFrame) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for knob_name in sensitivity_summary["knob_name"].unique():
        knob_frame = sensitivity_summary.loc[sensitivity_summary["knob_name"] == knob_name].sort_values("knob_scale")
        if len(knob_frame) < 2:
            continue
        baseline = knob_frame.iloc[0]
        raised = knob_frame.iloc[-1]
        if knob_name == "shock_scale":
            checks[knob_name] = bool(
                raised["shock_to_calm_abs_return_ratio_mean"] > baseline["shock_to_calm_abs_return_ratio_mean"]
            )
        elif knob_name == "meta_order_scale":
            checks[knob_name] = bool(raised["meta_directionality_mean"] > baseline["meta_directionality_mean"])
        elif knob_name == "excitation_scale":
            checks[knob_name] = bool(raised["event_clustering_mean"] > baseline["event_clustering_mean"])
        elif knob_name == "fair_price_vol_scale":
            checks[knob_name] = bool(raised["realized_vol_mean"] > baseline["realized_vol_mean"])
    return checks


def _preset_compare(frame: pd.DataFrame, left: str, right: str, column: str) -> bool:
    if left not in frame.index or right not in frame.index:
        return False
    return bool(frame.loc[left, column] > frame.loc[right, column])


def _ordered_preset_metric(frame: pd.DataFrame, column: str, *, order: Sequence[str]) -> bool:
    if any(label not in frame.index for label in order):
        return False
    values = [frame.loc[label, column] for label in order]
    return bool(all(left < right for left, right in zip(values, values[1:])))


def _safe_mean(values: pd.Series | Sequence[float]) -> float:
    series = pd.Series(values, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return float("nan")
    return float(finite.mean())


def _safe_std(values: pd.Series | Sequence[float]) -> float:
    series = pd.Series(values, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) < 2:
        return 0.0
    return float(finite.std(ddof=0))


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 2:
        return 0.0
    corr = float(paired["left"].corr(paired["right"]))
    if np.isnan(corr):
        return 0.0
    return corr


def _safe_autocorr(values: pd.Series | Sequence[float], lag: int) -> float:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) <= lag:
        return 0.0
    corr = float(series.autocorr(lag=lag))
    if np.isnan(corr):
        return 0.0
    return corr


def _coefficient_of_variation(mean_value: float, std_value: float) -> float:
    if not np.isfinite(mean_value) or abs(mean_value) <= EPSILON:
        return float("inf") if std_value > 0 else 0.0
    return float(abs(std_value) / abs(mean_value))
