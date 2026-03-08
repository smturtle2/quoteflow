from __future__ import annotations

"""Validation helpers for longer-form orderwave experiment sweeps."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
from time import perf_counter
import tracemalloc
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from orderwave.history import DEBUG_COLUMNS, EVENT_COLUMNS
from orderwave.market import Market

try:
    import psutil
except ImportError:  # pragma: no cover - exercised through fallback path
    psutil = None


PHASE_ORDER: tuple[str, ...] = ("open", "mid", "close")
DEFAULT_PRESETS: tuple[str, ...] = ("balanced", "trend", "volatile")
DEFAULT_SENSITIVITY_KNOBS: tuple[str, ...] = (
    "limit_rate_scale",
    "market_rate_scale",
    "cancel_rate_scale",
    "fair_price_vol_scale",
    "regime_transition_scale",
    "seasonality_scale",
    "excitation_scale",
    "meta_order_scale",
    "shock_scale",
)
DEFAULT_SENSITIVITY_SCALES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
CORE_SENSITIVITY_KNOBS: tuple[str, ...] = (
    "market_rate_scale",
    "cancel_rate_scale",
    "fair_price_vol_scale",
    "excitation_scale",
    "meta_order_scale",
    "shock_scale",
)
INVARIANT_FAILURE_COLUMNS: list[str] = [
    "preset",
    "seed",
    "stage",
    "step",
    "event_idx",
    "invariant_name",
    "details",
]
EPSILON = 1e-9
BASELINE_THROUGHPUT_FLOOR = {
    "balanced": 300.0,
    "trend": 225.0,
    "volatile": 200.0,
}
SOAK_PEAK_MEMORY_BUDGET_MB = {
    "balanced": 2_048.0,
    "trend": 3_072.0,
    "volatile": 3_584.0,
}
BYTES_PER_LOGGED_EVENT_BUDGET = 300.0
OPTIONAL_NUMERIC_COLUMNS = {"knob_scale", "repeat_idx"}
VALIDATION_BASELINE_SCHEMA_VERSION = 1
VALIDATION_BASELINE_METRIC_RULES: dict[str, dict[str, tuple[str, float]]] = {
    "baseline": {
        "mean_spread_mean": ("abs", 0.003),
        "realized_vol_mean": ("abs", 0.003),
        "abs_return_acf1_mean": ("abs", 0.08),
        "events_per_step_mean": ("abs", 8.0),
        "market_buy_share_mean": ("abs", 0.08),
    },
    "soak": {
        "peak_memory_mb_mean": ("max", 256.0),
        "bytes_per_logged_event_mean": ("max", 40.0),
    },
}


@dataclass(frozen=True)
class ValidationRun:
    """Container for a single simulated run and its output tables."""

    market: Market | None
    history: pd.DataFrame
    event_history: pd.DataFrame
    debug_history: pd.DataFrame
    start_snapshot: dict[str, Any] | None
    end_snapshot: dict[str, Any] | None
    elapsed_seconds: float
    peak_memory_mb: float
    peak_memory_increase_mb: float
    memory_metric: str
    memory_growth_without_recovery: bool
    requested_steps: int
    warmup_fraction: float
    warmup_cutoff_step: int
    run_failed: bool
    error_message: str | None


@dataclass(frozen=True)
class ValidationPipelineResult:
    """Collected outputs from the final validation pipeline."""

    run_metrics: pd.DataFrame
    preset_summary: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    invariant_failures: pd.DataFrame
    reproducibility: pd.DataFrame
    acceptance: dict[str, Any]
    diagnostics_paths: dict[str, Path]
    artifact_paths: dict[str, Path]


class _MemoryTracker:
    def __init__(self) -> None:
        self.metric_name = "rss_mb" if psutil is not None else "python_heap_mb"
        self._process = psutil.Process() if psutil is not None else None
        self._stop = threading.Event()
        self._samples: list[float] = []
        self._thread: threading.Thread | None = None
        self._peak_memory_mb = 0.0
        self._start_memory_mb = 0.0

    def __enter__(self) -> "_MemoryTracker":
        if self._process is not None:
            self._start_memory_mb = self._sample_rss()
            self._samples.append(self._start_memory_mb)
            self._thread = threading.Thread(target=self._rss_loop, daemon=True)
            self._thread.start()
        else:
            self._start_memory_mb = 0.0
            tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._process is not None:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            self._samples.append(self._sample_rss())
            self._peak_memory_mb = max(self._samples, default=0.0)
        else:
            current, peak = tracemalloc.get_traced_memory()
            self._peak_memory_mb = float(peak) / (1024.0 * 1024.0)
            tracemalloc.stop()

    @property
    def peak_memory_mb(self) -> float:
        return float(self._peak_memory_mb)

    @property
    def peak_memory_increase_mb(self) -> float:
        if self.metric_name == "rss_mb":
            return max(0.0, float(self._peak_memory_mb - self._start_memory_mb))
        return float(self._peak_memory_mb)

    @property
    def growth_without_recovery(self) -> bool:
        if self.metric_name != "rss_mb" or len(self._samples) < 8:
            return False
        samples = np.array(self._samples, dtype=float)
        anchor = samples[max(1, len(samples) // 4) :]
        if len(anchor) < 4:
            return False
        diffs = np.diff(anchor)
        mostly_nonnegative = float(np.mean(diffs >= -2.0)) >= 0.9
        terminal_growth = anchor[-1] >= (np.nanmin(anchor) * 1.25)
        return bool(mostly_nonnegative and terminal_growth)

    def _rss_loop(self) -> None:
        while not self._stop.wait(0.05):
            self._samples.append(self._sample_rss())

    def _sample_rss(self) -> float:
        assert self._process is not None
        return float(self._process.memory_info().rss) / (1024.0 * 1024.0)


def run_market_validation(
    *,
    preset: str,
    seed: int,
    steps: int,
    config_overrides: Mapping[str, object] | None = None,
    warmup_fraction: float = 0.10,
) -> ValidationRun:
    """Run one market simulation and capture its outputs."""

    config: dict[str, object] = {"preset": preset, "logging_mode": "full"}
    if config_overrides:
        config.update(config_overrides)

    market: Market | None = None
    start_snapshot: dict[str, Any] | None = None
    end_snapshot: dict[str, Any] | None = None
    history = pd.DataFrame()
    event_history = pd.DataFrame()
    debug_history = pd.DataFrame()
    error_message: str | None = None
    run_failed = False
    elapsed = 0.0

    tracker = _MemoryTracker()
    try:
        market = Market(seed=seed, config=config)
        start_snapshot = market.get()
        with tracker:
            started = perf_counter()
            market.gen(steps=steps)
            elapsed = perf_counter() - started
        end_snapshot = market.get()
        history = market.get_history()
        if market.config.logging_mode == "full":
            event_history = market.get_event_history()
            debug_history = market.get_debug_history()
        else:
            event_history = pd.DataFrame(columns=EVENT_COLUMNS)
            debug_history = pd.DataFrame(columns=DEBUG_COLUMNS)
    except Exception as exc:  # pragma: no cover - exercised via failure paths
        run_failed = True
        error_message = f"{type(exc).__name__}: {exc}"
        if market is not None:
            try:
                end_snapshot = market.get()
                history = market.get_history()
                if market.config.logging_mode == "full":
                    event_history = market.get_event_history()
                    debug_history = market.get_debug_history()
                else:
                    event_history = pd.DataFrame(columns=EVENT_COLUMNS)
                    debug_history = pd.DataFrame(columns=DEBUG_COLUMNS)
            except Exception:
                history = pd.DataFrame()
                event_history = pd.DataFrame()
                debug_history = pd.DataFrame()
    warmup_cutoff = int(max(0, min(steps, math.floor(float(steps) * float(warmup_fraction)))))

    return ValidationRun(
        market=market,
        history=history,
        event_history=event_history,
        debug_history=debug_history,
        start_snapshot=start_snapshot,
        end_snapshot=end_snapshot,
        elapsed_seconds=float(elapsed),
        peak_memory_mb=tracker.peak_memory_mb,
        peak_memory_increase_mb=tracker.peak_memory_increase_mb,
        memory_metric=tracker.metric_name,
        memory_growth_without_recovery=tracker.growth_without_recovery,
        requested_steps=int(steps),
        warmup_fraction=float(warmup_fraction),
        warmup_cutoff_step=int(warmup_cutoff),
        run_failed=run_failed,
        error_message=error_message,
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

    step_index = _analysis_step_index(run)
    history = _analysis_history(run)
    events = _analysis_events(run)
    debug = _analysis_debug(run)
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

    phase_metrics = _phase_metrics(history, events, market_events)
    joined = _join_events_and_debug(events, debug)
    shock_abs_return, calm_abs_return = _shock_vs_calm_abs_return(history, debug)
    meta_active_ratio, meta_inactive_ratio = _meta_directional_ratios(joined)

    crossing_history = int((run.history["best_bid"] >= run.history["best_ask"]).sum()) if not run.history.empty else 0
    crossing_events = int((run.event_history["best_bid_after"] >= run.event_history["best_ask_after"]).sum()) if not run.event_history.empty else 0
    realized_vol = _safe_mean(history["realized_vol"]) if not history.empty else 0.0
    if realized_vol <= 0.0:
        realized_vol = _safe_std(mid_return)

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
        "mean_return": _safe_mean(mid_return),
        "return_std": _safe_std(mid_return),
        "realized_vol": float(realized_vol),
        "abs_return_acf1": _safe_autocorr(abs_return, lag=1),
        "abs_return_acf5": _safe_autocorr(abs_return, lag=5),
        "mean_spread": _safe_mean(history["spread"]) if not history.empty else 0.0,
        "median_spread": float(history["spread"].median()) if not history.empty else 0.0,
        "spread_unique_count": int(history["spread"].nunique()) if not history.empty else 0,
        "spread_gt_min_tick_ratio": float((history["spread"] > _tick_size_from_run(run)).mean()) if not history.empty else 0.0,
        "crossing_violation_count": int(crossing_history + crossing_events),
        "mean_depth_imbalance": _safe_mean(history["depth_imbalance"]) if not history.empty else 0.0,
        "depth_imbalance_std": _safe_std(history["depth_imbalance"]) if not history.empty else 0.0,
        "imbalance_next_mid_return_corr": _safe_corr(history["depth_imbalance"], next_mid_return) if not history.empty else 0.0,
        "events_per_step": float(event_count / effective_steps),
        "events_per_step_market": float(market_event_count / effective_steps),
        "market_buy_share": float(buy_count / max(market_event_count, 1)),
        "market_sell_share": float(sell_count / max(market_event_count, 1)),
        "cancel_share": float(len(cancel_events) / max(event_count, 1)),
        "limit_share": float(len(limit_events) / max(event_count, 1)),
        "buy_event_count_acf1": _safe_autocorr(_per_step_event_count(step_index, market_events.loc[market_events["side"] == "buy"]), lag=1),
        "cancel_event_count_acf1": _safe_autocorr(_per_step_event_count(step_index, cancel_events), lag=1),
        "trade_sign_acf1": _safe_autocorr(market_signs, lag=1),
        "directional_ratio": _directional_ratio(market_signs),
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
        "regime_switch_rate": _regime_switch_rate(history),
        "steps_per_second": float(run.requested_steps / max(run.elapsed_seconds, EPSILON)),
        "events_logged_per_second": float(logged_event_count / max(run.elapsed_seconds, EPSILON)),
        "peak_memory_mb": float(run.peak_memory_mb),
        "bytes_per_logged_event": float((run.peak_memory_mb * 1024.0 * 1024.0) / max(logged_event_count, 1)),
        "memory_metric": run.memory_metric,
        "memory_growth_without_recovery": bool(run.memory_growth_without_recovery),
        "run_failed": bool(run.run_failed),
        "error_message": run.error_message or "",
    }


def run_validation_grid(
    *,
    presets: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    warmup_fraction: float = 0.10,
    config_overrides_by_preset: Mapping[str, Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Run the baseline preset grid and return one row per preset/seed."""

    metrics, _ = _execute_run_grid(
        stage="baseline",
        presets=presets,
        seeds=seeds,
        steps=steps,
        warmup_fraction=warmup_fraction,
        config_overrides_by_preset=config_overrides_by_preset,
    )
    return metrics


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
                row[f"{metric}_mean"] = _safe_mean(frame[metric])
                row[f"{metric}_std"] = _safe_std(frame[metric])
            else:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_std"] = 0.0
        for metric in cv_metrics:
            row[f"{metric}_cv"] = _coefficient_of_variation(row[f"{metric}_mean"], row[f"{metric}_std"])
        rows.append(row)
    return pd.DataFrame(rows)


def run_reproducibility_checks(
    *,
    presets: Sequence[str],
    seed: int = 42,
    steps: int = 250,
    warmup_fraction: float = 0.10,
) -> pd.DataFrame:
    """Check same-seed reproducibility and ``gen`` vs repeated ``step`` consistency."""

    rows: list[dict[str, Any]] = []
    for preset in presets:
        repeated_runs = [
            run_market_validation(
                preset=preset,
                seed=seed,
                steps=steps,
                warmup_fraction=warmup_fraction,
            )
            for _ in range(3)
        ]
        step_market = Market(seed=seed, config={"preset": preset})
        for _ in range(steps):
            step_market.step()

        repeated_hashes = []
        metric_hashes = []
        for run in repeated_runs:
            metrics = compute_run_metrics(
                run,
                stage="reproducibility",
                preset=preset,
                seed=seed,
                config_label="repeat",
            )
            repeated_hashes.append(
                {
                    "history": _stable_frame_hash(run.history),
                    "events": _stable_frame_hash(run.event_history),
                    "debug": _stable_frame_hash(run.debug_history),
                }
            )
            metric_hashes.append(_stable_object_hash(_deterministic_metric_view(metrics)))

        base_hash = repeated_hashes[0]
        history_hash_equal = all(item["history"] == base_hash["history"] for item in repeated_hashes[1:])
        event_hash_equal = all(item["events"] == base_hash["events"] for item in repeated_hashes[1:])
        debug_hash_equal = all(item["debug"] == base_hash["debug"] for item in repeated_hashes[1:])
        metrics_hash_equal = all(metric_hash == metric_hashes[0] for metric_hash in metric_hashes[1:])

        gen_history = repeated_runs[0].history
        gen_events = repeated_runs[0].event_history
        gen_debug = repeated_runs[0].debug_history
        step_history = step_market.get_history()
        step_events = step_market.get_event_history()
        step_debug = step_market.get_debug_history()

        gen_vs_step_history = gen_history.equals(step_history)
        gen_vs_step_events = gen_events.equals(step_events)
        gen_vs_step_debug = gen_debug.equals(step_debug)

        rows.append(
            {
                "preset": preset,
                "seed": int(seed),
                "steps": int(steps),
                "history_hash_equal": bool(history_hash_equal),
                "event_hash_equal": bool(event_hash_equal),
                "debug_hash_equal": bool(debug_hash_equal),
                "metrics_hash_equal": bool(metrics_hash_equal),
                "gen_vs_step_history_equal": bool(gen_vs_step_history),
                "gen_vs_step_event_equal": bool(gen_vs_step_events),
                "gen_vs_step_debug_equal": bool(gen_vs_step_debug),
                "all_reproducible": bool(
                    history_hash_equal
                    and event_hash_equal
                    and debug_hash_equal
                    and metrics_hash_equal
                    and gen_vs_step_history
                    and gen_vs_step_events
                    and gen_vs_step_debug
                ),
            }
        )
    return pd.DataFrame(rows)


def run_sensitivity_grid(
    *,
    preset: str,
    seeds: Sequence[int],
    steps: int,
    warmup_fraction: float = 0.10,
    knobs: Sequence[str] = DEFAULT_SENSITIVITY_KNOBS,
    scales: Sequence[float] = DEFAULT_SENSITIVITY_SCALES,
) -> pd.DataFrame:
    """Run knob sensitivity experiments for one preset."""

    metrics, _ = _execute_sensitivity_grid(
        preset=preset,
        seeds=seeds,
        steps=steps,
        warmup_fraction=warmup_fraction,
        knobs=knobs,
        scales=scales,
    )
    return metrics


def benchmark_logging_modes(
    *,
    preset: str = "balanced",
    seed: int = 42,
    steps: int = 10_000,
    warmup_fraction: float = 0.10,
) -> pd.DataFrame:
    """Compare full logging against history-only mode on one configuration."""

    rows: list[dict[str, Any]] = []
    for logging_mode in ("full", "history_only"):
        run = run_market_validation(
            preset=preset,
            seed=seed,
            steps=steps,
            config_overrides={"logging_mode": logging_mode},
            warmup_fraction=warmup_fraction,
        )
        logged_rows = len(run.event_history) + len(run.debug_history)
        rows.append(
            {
                "preset": preset,
                "seed": int(seed),
                "steps": int(steps),
                "logging_mode": logging_mode,
                "steps_per_second": float(steps / max(run.elapsed_seconds, EPSILON)),
                "peak_memory_mb": float(run.peak_memory_mb),
                "peak_memory_increase_mb": float(run.peak_memory_increase_mb),
                "logged_rows": int(logged_rows),
                "bytes_per_logged_event": float((run.peak_memory_mb * 1024.0 * 1024.0) / max(logged_rows, 1)),
                "run_failed": bool(run.run_failed),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) == 2:
        full_row = frame.loc[frame["logging_mode"] == "full"].iloc[0]
        history_only_row = frame.loc[frame["logging_mode"] == "history_only"].iloc[0]
        frame["peak_memory_reduction_pct_vs_full"] = np.nan
        frame["throughput_improvement_pct_vs_full"] = np.nan
        frame.loc[frame["logging_mode"] == "history_only", "peak_memory_reduction_pct_vs_full"] = (
            100.0
            * (float(full_row["peak_memory_increase_mb"]) - float(history_only_row["peak_memory_increase_mb"]))
            / max(float(full_row["peak_memory_increase_mb"]), EPSILON)
        )
        frame.loc[frame["logging_mode"] == "history_only", "throughput_improvement_pct_vs_full"] = (
            100.0 * (float(history_only_row["steps_per_second"]) - float(full_row["steps_per_second"])) / max(float(full_row["steps_per_second"]), EPSILON)
    )
    return frame


def measure_performance(
    *,
    preset: str = "balanced",
    seeds: Sequence[int] = (1,),
    steps: int = 20_000,
    warmup_fraction: float = 0.10,
    throughput_floor: float | None = None,
    logging_compare_seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the canonical performance sweep for one preset.

    Returns a dictionary with:

    - ``seed_metrics``: per-seed throughput and memory metrics
    - ``summary``: one-row aggregate summary including floor status
    - ``logging_compare``: full vs history-only comparison for one seed
    """

    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("seeds must not be empty")

    rows: list[dict[str, Any]] = []
    for seed in seed_list:
        run = run_market_validation(
            preset=preset,
            seed=seed,
            steps=steps,
            warmup_fraction=warmup_fraction,
        )
        metrics = compute_run_metrics(
            run,
            stage="performance",
            preset=preset,
            seed=seed,
            config_label="performance",
        )
        rows.append(
            {
                "preset": preset,
                "seed": int(seed),
                "steps": int(steps),
                "steps_per_second": float(metrics["steps_per_second"]),
                "events_per_step": float(metrics["events_per_step"]),
                "peak_memory_mb": float(metrics["peak_memory_mb"]),
                "bytes_per_logged_event": float(metrics["bytes_per_logged_event"]),
                "run_failed": bool(metrics["run_failed"]),
            }
        )

    seed_metrics = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    floor = float(throughput_floor) if throughput_floor is not None else float(BASELINE_THROUGHPUT_FLOOR.get(preset, 0.0))
    failed_runs = int(seed_metrics["run_failed"].sum())
    summary = pd.DataFrame(
        [
            {
                "preset": preset,
                "steps": int(steps),
                "seeds": int(len(seed_list)),
                "throughput_floor": float(floor),
                "mean_steps_per_second": float(seed_metrics["steps_per_second"].mean()),
                "min_steps_per_second": float(seed_metrics["steps_per_second"].min()),
                "max_steps_per_second": float(seed_metrics["steps_per_second"].max()),
                "mean_events_per_step": float(seed_metrics["events_per_step"].mean()),
                "mean_peak_memory_mb": float(seed_metrics["peak_memory_mb"].mean()),
                "mean_bytes_per_logged_event": float(seed_metrics["bytes_per_logged_event"].mean()),
                "failed_runs": failed_runs,
                "floor_pass": bool(failed_runs == 0 and float(seed_metrics["steps_per_second"].mean()) >= floor),
            }
        ]
    )
    compare_seed = int(logging_compare_seed) if logging_compare_seed is not None else seed_list[0]
    logging_compare = benchmark_logging_modes(
        preset=preset,
        seed=compare_seed,
        steps=steps,
        warmup_fraction=warmup_fraction,
    )
    return {
        "seed_metrics": seed_metrics,
        "summary": summary,
        "logging_compare": logging_compare,
    }


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
            "limit_share_mean": _safe_mean(frame["limit_share"]),
            "events_per_step_market_mean": _safe_mean(frame["events_per_step_market"]),
            "cancel_share_mean": _safe_mean(frame["cancel_share"]),
            "realized_vol_mean": _safe_mean(frame["realized_vol"]),
            "regime_switch_rate_mean": _safe_mean(frame["regime_switch_rate"]),
            "phase_structure_signal_mean": _safe_mean(frame["phase_spread_range"] + frame["phase_fill_range"]),
            "event_clustering_mean": _safe_mean((frame["buy_event_count_acf1"] + frame["cancel_event_count_acf1"]) / 2.0),
            "meta_active_directional_ratio_mean": _safe_mean(frame["meta_active_directional_ratio"]),
            "shock_to_calm_ratio_mean": _safe_mean(frame["shock_to_calm_ratio"]),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    direction_rows: list[dict[str, Any]] = []
    for knob_name, frame in summary.groupby("knob_name", sort=False):
        ordered = frame.sort_values("knob_scale").reset_index(drop=True)
        target_metric = _sensitivity_target_metric(knob_name)
        values = ordered[target_metric].to_numpy(dtype=float)
        direction_ok = _quasi_monotonic(values)
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


def evaluate_validation_results(
    *,
    run_metrics: pd.DataFrame,
    preset_summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    invariant_failures: pd.DataFrame,
) -> dict[str, Any]:
    """Produce the final GO / CONDITIONAL / NO-GO verdict."""

    baseline_metrics = run_metrics.loc[run_metrics["stage"] == "baseline"].copy()
    soak_metrics = run_metrics.loc[run_metrics["stage"] == "soak"].copy()
    baseline_summary = preset_summary.loc[preset_summary["stage"] == "baseline"].set_index("preset")
    soak_summary = preset_summary.loc[preset_summary["stage"] == "soak"].set_index("preset")

    invariants_ok = invariant_failures.empty
    reproducible_ok = bool(not reproducibility.empty and reproducibility["all_reproducible"].all())
    performance_checks: dict[str, bool] = {
        "soak_failures": bool(not soak_metrics.empty and (~soak_metrics["run_failed"]).all()),
    }
    for preset, floor in BASELINE_THROUGHPUT_FLOOR.items():
        performance_checks[f"{preset}_throughput_floor"] = bool(
            preset in baseline_summary.index and float(baseline_summary.loc[preset, "steps_per_second_mean"]) >= floor
        )
    for preset, budget in SOAK_PEAK_MEMORY_BUDGET_MB.items():
        performance_checks[f"{preset}_peak_rss_budget"] = bool(
            preset in soak_summary.index and float(soak_summary.loc[preset, "peak_memory_mb_mean"]) < budget
        )
    performance_checks["soak_bytes_per_logged_event_budget"] = bool(
        not soak_summary.empty and (soak_summary["bytes_per_logged_event_mean"] < BYTES_PER_LOGGED_EVENT_BUDGET).all()
    )
    performance_ok = all(performance_checks.values())

    classifier_accuracy = _leave_one_out_centroid_accuracy(
        baseline_metrics,
        features=("realized_vol", "mean_spread", "trade_sign_acf1", "events_per_step"),
    )
    preset_checks = {
        "volatile_realized_vol_gt_balanced": _preset_compare(baseline_summary, "volatile", "balanced", "realized_vol_mean"),
        "volatile_mean_spread_ge_balanced": _preset_compare(
            baseline_summary,
            "volatile",
            "balanced",
            "mean_spread_mean",
            comparison="ge",
        ),
        "trend_trade_sign_acf1_gt_balanced": _preset_compare(
            baseline_summary,
            "trend",
            "balanced",
            "trade_sign_acf1_mean",
        ),
        "preset_classifier_accuracy": classifier_accuracy >= 0.75,
    }
    preset_separation_ok = all(preset_checks.values())

    stylized_checks = _evaluate_stylized_facts(baseline_summary)
    mandatory_stylized = (
        stylized_checks["abs_return_clustering"],
        stylized_checks["spread_variation"],
        stylized_checks["shock_response"],
    )
    stylized_ok = bool(sum(bool(value) for value in stylized_checks.values()) >= 5 and all(mandatory_stylized))

    sensitivity_checks = _evaluate_sensitivity_summary(sensitivity_summary)
    core_passes = sum(int(sensitivity_checks.get(knob, False)) for knob in CORE_SENSITIVITY_KNOBS)
    total_passes = sum(int(value) for value in sensitivity_checks.values())
    sensitivity_ok = bool(core_passes >= 4 and total_passes >= 6)

    seed_stability = _evaluate_seed_stability(baseline_metrics, baseline_summary)
    seed_stability_ok = bool(seed_stability["passes"])

    hard_gates_ok = invariants_ok and reproducible_ok and performance_ok
    soft_gates_ok = preset_separation_ok and stylized_ok and sensitivity_ok and seed_stability_ok
    if not hard_gates_ok:
        decision = "NO-GO"
        suitability = "부적합"
    elif soft_gates_ok:
        decision = "GO"
        suitability = "적합"
    else:
        decision = "CONDITIONAL"
        suitability = "조건부 적합"

    strengths: list[str] = []
    weaknesses: list[str] = []
    if invariants_ok:
        strengths.append("구조적 불변식 위반이 없고 로그 정합성이 유지됨")
    else:
        weaknesses.append("구조적 불변식 또는 summary non-finite 문제가 존재함")
    if reproducible_ok:
        strengths.append("동일 seed 재실행과 gen/step 경로가 일치함")
    else:
        weaknesses.append("재현성 검사가 실패함")
    if preset_separation_ok:
        strengths.append("preset별 경로 특성이 핵심 지표 공간에서 분리됨")
    else:
        weaknesses.append("preset 분리가 약하거나 classifier 분리도가 부족함")
    if stylized_ok:
        strengths.append("변동성 군집, spread variation, shock 반응 등 시간 구조가 관찰됨")
    else:
        weaknesses.append("stylized fact 또는 시간 구조가 충분히 드러나지 않음")
    if sensitivity_ok:
        strengths.append("노출된 knob가 대체로 의도한 방향으로 출력 분포를 제어함")
    else:
        weaknesses.append("민감도 반응이 약하거나 일부 knob 방향성이 뒤집힘")
    if seed_stability_ok:
        strengths.append("seed 변화에도 preset-level 결론이 유지됨")
    else:
        weaknesses.append("seed 민감도가 높아 정성적 결론이 흔들림")
    if performance_ok:
        strengths.append("반복 실험용 처리량과 장기 soak 안정성이 확보됨")
    else:
        weaknesses.append("성능 floor 또는 장기 soak 메모리 예산을 넘김")

    immediate_scope = (
        "preset 비교, multi-seed 실험, agent sandbox, stylized-state 연구"
        if decision != "NO-GO"
        else "추가 수정 전에는 연구용 채택 범위를 권장하지 않음"
    )
    required_fix = (
        "없음"
        if decision == "GO"
        else ", ".join(weaknesses[:2]) if weaknesses else "추가 검증 필요"
    )

    return {
        "decision": decision,
        "suitability": suitability,
        "invariants_ok": invariants_ok,
        "reproducibility_ok": reproducible_ok,
        "performance_ok": performance_ok,
        "preset_separation_ok": preset_separation_ok,
        "stylized_facts_ok": stylized_ok,
        "sensitivity_ok": sensitivity_ok,
        "seed_stability_ok": seed_stability_ok,
        "performance_checks": performance_checks,
        "preset_checks": {**preset_checks, "classifier_accuracy": float(classifier_accuracy)},
        "stylized_checks": stylized_checks,
        "sensitivity_checks": sensitivity_checks,
        "seed_stability": seed_stability,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "immediate_scope": immediate_scope,
        "required_fix": required_fix,
        "conclusion_market_state": (
            f"이 엔진은 execution simulator로 보지 않고 synthetic market-state generator로 볼 때 {suitability}."
        ),
        "conclusion_scope": f"현재 가장 신뢰할 수 있는 사용처는 {immediate_scope}.",
        "conclusion_required_fix": f"채택 전 반드시 보완할 항목은 {required_fix}.",
    }


def run_validation_pipeline(
    *,
    outdir: Path,
    presets: Sequence[str] = DEFAULT_PRESETS,
    baseline_seeds: int = 20,
    baseline_steps: int = 20_000,
    sensitivity_seeds: int = 8,
    sensitivity_steps: int = 15_000,
    long_run_seeds: int = 3,
    long_run_steps: int = 200_000,
    seed_start: int = 1,
    warmup_fraction: float = 0.10,
    diagnostics_seed_policy: str = "median-realized-vol",
    sensitivity_knobs: Sequence[str] = DEFAULT_SENSITIVITY_KNOBS,
    sensitivity_scales: Sequence[float] = DEFAULT_SENSITIVITY_SCALES,
    jobs: int = 1,
) -> ValidationPipelineResult:
    """Run the full final validation pipeline and write repo-standard artifacts."""

    outdir.mkdir(parents=True, exist_ok=True)
    presets = tuple(presets)
    jobs = max(1, int(jobs))
    baseline_seed_list = list(range(seed_start, seed_start + baseline_seeds))
    sensitivity_seed_list = baseline_seed_list[: max(1, min(sensitivity_seeds, len(baseline_seed_list)))]
    soak_seed_list = baseline_seed_list[: max(1, min(long_run_seeds, len(baseline_seed_list)))]

    baseline_metrics, baseline_failures = _execute_run_grid(
        stage="baseline",
        presets=presets,
        seeds=baseline_seed_list,
        steps=baseline_steps,
        warmup_fraction=warmup_fraction,
        jobs=jobs,
    )
    sensitivity_metrics, sensitivity_failures = _execute_sensitivity_grid(
        preset="balanced",
        seeds=sensitivity_seed_list,
        steps=sensitivity_steps,
        warmup_fraction=warmup_fraction,
        knobs=sensitivity_knobs,
        scales=sensitivity_scales,
        jobs=jobs,
    )
    soak_metrics, soak_failures = _execute_run_grid(
        stage="soak",
        presets=presets,
        seeds=soak_seed_list,
        steps=long_run_steps,
        warmup_fraction=warmup_fraction,
        jobs=jobs,
    )
    reproducibility = run_reproducibility_checks(
        presets=presets,
        seed=baseline_seed_list[0],
        steps=min(500, max(50, baseline_steps // 20)),
        warmup_fraction=warmup_fraction,
    )

    run_metric_frames = [frame for frame in (baseline_metrics, sensitivity_metrics, soak_metrics) if not frame.empty]
    if run_metric_frames:
        run_metric_columns = list(dict.fromkeys(column for frame in run_metric_frames for column in frame.columns))
        concat_ready = [frame.dropna(axis=1, how="all") for frame in run_metric_frames]
        run_metrics = pd.concat(concat_ready, ignore_index=True, sort=False).reindex(columns=run_metric_columns)
    else:
        run_metrics = pd.DataFrame()
    preset_frames = [frame for frame in (baseline_metrics, soak_metrics) if not frame.empty]
    preset_summary = summarize_validation_grid(
        pd.concat(preset_frames, ignore_index=True, sort=False) if preset_frames else pd.DataFrame()
    )
    sensitivity_summary = summarize_sensitivity_grid(sensitivity_metrics)

    invariant_failures = pd.concat(
        [
            baseline_failures,
            sensitivity_failures,
            soak_failures,
            _reproducibility_failures(reproducibility),
        ],
        ignore_index=True,
        sort=False,
    )
    invariant_failures = pd.concat(
        [
            invariant_failures,
            _table_nonfinite_failures(run_metrics, stage="run_metrics"),
            _table_nonfinite_failures(preset_summary, stage="preset_summary"),
            _table_nonfinite_failures(sensitivity_summary, stage="sensitivity_summary"),
        ],
        ignore_index=True,
        sort=False,
    )
    invariant_failures = invariant_failures.reindex(columns=INVARIANT_FAILURE_COLUMNS).fillna({"details": ""})

    acceptance = evaluate_validation_results(
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        reproducibility=reproducibility,
        sensitivity_summary=sensitivity_summary,
        invariant_failures=invariant_failures,
    )

    diagnostics_seeds = _select_diagnostics_seeds(
        baseline_metrics,
        presets=presets,
        policy=diagnostics_seed_policy,
    )
    diagnostics_paths: dict[str, Path] = {}
    for preset, seed in diagnostics_seeds.items():
        run = run_market_validation(
            preset=preset,
            seed=seed,
            steps=baseline_steps,
            warmup_fraction=warmup_fraction,
        )
        if run.market is None:
            continue
        figure = run.market.plot_diagnostics(title=f"{preset} diagnostics ({seed})", figsize=(14, 8.5))
        path = outdir / f"diagnostics_{preset}_{seed}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        diagnostics_paths[preset] = path
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    artifact_paths = {
        "validation_summary": outdir / "validation_summary.md",
        "run_metrics": outdir / "run_metrics.csv",
        "preset_summary": outdir / "preset_summary.csv",
        "sensitivity_summary": outdir / "sensitivity_summary.csv",
        "invariant_failures": outdir / "invariant_failures.csv",
        "acceptance_decision": outdir / "acceptance_decision.md",
    }
    run_metrics.to_csv(artifact_paths["run_metrics"], index=False)
    preset_summary.to_csv(artifact_paths["preset_summary"], index=False)
    sensitivity_summary.to_csv(artifact_paths["sensitivity_summary"], index=False)
    invariant_failures.to_csv(artifact_paths["invariant_failures"], index=False)
    _write_validation_summary(
        outpath=artifact_paths["validation_summary"],
        presets=presets,
        baseline_seed_list=baseline_seed_list,
        sensitivity_seed_list=sensitivity_seed_list,
        soak_seed_list=soak_seed_list,
        baseline_steps=baseline_steps,
        sensitivity_steps=sensitivity_steps,
        long_run_steps=long_run_steps,
        warmup_fraction=warmup_fraction,
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        sensitivity_summary=sensitivity_summary,
        reproducibility=reproducibility,
        invariant_failures=invariant_failures,
        acceptance=acceptance,
        diagnostics_paths=diagnostics_paths,
    )
    _write_acceptance_decision(
        outpath=artifact_paths["acceptance_decision"],
        acceptance=acceptance,
    )

    return ValidationPipelineResult(
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        sensitivity_summary=sensitivity_summary,
        invariant_failures=invariant_failures,
        reproducibility=reproducibility,
        acceptance=acceptance,
        diagnostics_paths=diagnostics_paths,
        artifact_paths=artifact_paths,
    )


def extract_validation_baseline(result: ValidationPipelineResult) -> dict[str, Any]:
    """Extract a compact golden baseline from a validation result."""

    preset_index = result.preset_summary.set_index(["stage", "preset"])
    metrics: dict[str, dict[str, dict[str, dict[str, float | str]]]] = {}
    for stage, metric_rules in VALIDATION_BASELINE_METRIC_RULES.items():
        stage_rows: dict[str, dict[str, dict[str, float | str]]] = {}
        for preset in DEFAULT_PRESETS:
            if (stage, preset) not in preset_index.index:
                continue
            row = preset_index.loc[(stage, preset)]
            stage_rows[preset] = {
                metric_name: {
                    "value": float(row[metric_name]),
                    "mode": mode,
                    "tolerance": float(tolerance),
                }
                for metric_name, (mode, tolerance) in metric_rules.items()
            }
        metrics[stage] = stage_rows

    sensitivity_by_knob = (
        result.sensitivity_summary.groupby("knob_name", sort=False)["direction_ok"].max().to_dict()
        if not result.sensitivity_summary.empty
        else {}
    )
    sensitivity_by_knob = {str(knob): bool(value) for knob, value in sensitivity_by_knob.items()}

    acceptance = result.acceptance
    return {
        "schema_version": VALIDATION_BASELINE_SCHEMA_VERSION,
        "liquidity_backstop_default": "always",
        "acceptance": {
            "decision": str(acceptance["decision"]),
            "invariants_ok": bool(acceptance["invariants_ok"]),
            "reproducibility_ok": bool(acceptance["reproducibility_ok"]),
            "performance_ok": bool(acceptance["performance_ok"]),
            "preset_separation_ok": bool(acceptance["preset_separation_ok"]),
            "stylized_facts_ok": bool(acceptance["stylized_facts_ok"]),
            "sensitivity_ok": bool(acceptance["sensitivity_ok"]),
            "seed_stability_ok": bool(acceptance["seed_stability_ok"]),
            "performance_checks": {str(key): bool(value) for key, value in acceptance["performance_checks"].items()},
            "preset_checks": {
                str(key): (float(value) if key == "classifier_accuracy" else bool(value))
                for key, value in acceptance["preset_checks"].items()
            },
            "stylized_checks": {str(key): bool(value) for key, value in acceptance["stylized_checks"].items()},
            "sensitivity_checks": sensitivity_by_knob,
        },
        "metrics": metrics,
        "next_focus": "finer_event_feedback",
    }


def write_validation_baseline(path: Path, result: ValidationPipelineResult) -> None:
    baseline = extract_validation_baseline(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_validation_baseline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_validation_baseline(
    result: ValidationPipelineResult,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare a validation run against a stored golden baseline."""

    actual = extract_validation_baseline(result)
    failures: list[str] = []

    baseline_acceptance = baseline.get("acceptance", {})
    actual_acceptance = actual["acceptance"]
    exact_acceptance_fields = (
        "decision",
        "invariants_ok",
        "reproducibility_ok",
        "performance_ok",
        "preset_separation_ok",
        "stylized_facts_ok",
        "sensitivity_ok",
        "seed_stability_ok",
    )
    for field in exact_acceptance_fields:
        if actual_acceptance.get(field) != baseline_acceptance.get(field):
            failures.append(f"acceptance.{field}: expected {baseline_acceptance.get(field)!r}, got {actual_acceptance.get(field)!r}")

    for section_name in ("performance_checks", "stylized_checks", "sensitivity_checks"):
        expected_section = baseline_acceptance.get(section_name, {})
        actual_section = actual_acceptance.get(section_name, {})
        for key, expected in expected_section.items():
            actual_value = actual_section.get(key)
            if actual_value != expected:
                failures.append(f"acceptance.{section_name}.{key}: expected {expected!r}, got {actual_value!r}")

    expected_preset_checks = baseline_acceptance.get("preset_checks", {})
    actual_preset_checks = actual_acceptance.get("preset_checks", {})
    for key, expected in expected_preset_checks.items():
        actual_value = actual_preset_checks.get(key)
        if key == "classifier_accuracy":
            if float(actual_value) + 1e-9 < float(expected):
                failures.append(f"acceptance.preset_checks.{key}: expected >= {float(expected):.4f}, got {float(actual_value):.4f}")
        elif actual_value != expected:
            failures.append(f"acceptance.preset_checks.{key}: expected {expected!r}, got {actual_value!r}")

    for stage, stage_metrics in baseline.get("metrics", {}).items():
        actual_stage_metrics = actual["metrics"].get(stage, {})
        for preset, preset_metrics in stage_metrics.items():
            actual_preset_metrics = actual_stage_metrics.get(preset)
            if actual_preset_metrics is None:
                failures.append(f"metrics.{stage}.{preset}: missing preset metrics")
                continue
            for metric_name, expected in preset_metrics.items():
                actual_entry = actual_preset_metrics.get(metric_name)
                if actual_entry is None:
                    failures.append(f"metrics.{stage}.{preset}.{metric_name}: missing metric")
                    continue
                expected_value = float(expected["value"])
                tolerance = float(expected["tolerance"])
                mode = str(expected["mode"])
                actual_value = float(actual_entry["value"])
                if mode == "abs":
                    ok = abs(actual_value - expected_value) <= tolerance
                elif mode == "min":
                    ok = actual_value >= (expected_value - tolerance)
                elif mode == "max":
                    ok = actual_value <= (expected_value + tolerance)
                else:  # pragma: no cover - baseline schema guard
                    raise ValueError(f"unsupported validation baseline mode: {mode}")
                if not ok:
                    failures.append(
                        f"metrics.{stage}.{preset}.{metric_name}: expected {mode} {expected_value:.6f} +/- {tolerance:.6f}, got {actual_value:.6f}"
                    )

    if actual.get("liquidity_backstop_default") != baseline.get("liquidity_backstop_default"):
        failures.append(
            "liquidity_backstop_default: "
            f"expected {baseline.get('liquidity_backstop_default')!r}, got {actual.get('liquidity_backstop_default')!r}"
        )

    return {
        "matches": not failures,
        "failures": failures,
        "actual": actual,
    }


def _execute_run_grid(
    *,
    stage: str,
    presets: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    warmup_fraction: float,
    config_overrides_by_preset: Mapping[str, Mapping[str, object]] | None = None,
    jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks: list[dict[str, Any]] = []
    for preset in presets:
        preset_overrides = (config_overrides_by_preset or {}).get(preset, {})
        for seed in seeds:
            tasks.append(
                {
                    "stage": stage,
                    "preset": preset,
                    "seed": int(seed),
                    "steps": steps,
                    "warmup_fraction": warmup_fraction,
                    "config_overrides": dict(preset_overrides),
                    "config_label": stage,
                    "knob_name": None,
                    "knob_scale": None,
                    "repeat_idx": None,
                }
            )
    return _execute_tasks(tasks, jobs=jobs)


def _execute_sensitivity_grid(
    *,
    preset: str,
    seeds: Sequence[int],
    steps: int,
    warmup_fraction: float,
    knobs: Sequence[str],
    scales: Sequence[float],
    jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks: list[dict[str, Any]] = []
    for knob_name in knobs:
        for knob_scale in scales:
            for seed in seeds:
                tasks.append(
                    {
                        "stage": "sensitivity",
                        "preset": preset,
                        "seed": int(seed),
                        "steps": steps,
                        "warmup_fraction": warmup_fraction,
                        "config_overrides": {knob_name: float(knob_scale)},
                        "config_label": f"{knob_name}={knob_scale:.2f}",
                        "knob_name": knob_name,
                        "knob_scale": float(knob_scale),
                        "repeat_idx": None,
                    }
                )
    return _execute_tasks(tasks, jobs=jobs)


def _execute_tasks(tasks: Sequence[dict[str, Any]], *, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tasks:
        return pd.DataFrame(), pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    if jobs <= 0:
        results = [_run_validation_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(jobs), max_tasks_per_child=1) as executor:
            results = list(executor.map(_run_validation_task, tasks))
    rows = [metrics for metrics, _ in results]
    failure_rows = [pd.DataFrame.from_records(records, columns=INVARIANT_FAILURE_COLUMNS) for _, records in results]
    metrics_frame = pd.DataFrame(rows)
    sort_columns = [column for column in ("stage", "preset", "config_label", "seed") if column in metrics_frame.columns]
    if sort_columns:
        metrics_frame = metrics_frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    failures_frame = _concat_failures(failure_rows)
    if not failures_frame.empty:
        failures_frame = failures_frame.sort_values(
            ["stage", "preset", "seed", "step", "event_idx", "invariant_name"],
            kind="stable",
        ).reset_index(drop=True)
    return metrics_frame, failures_frame


def _run_validation_task(task: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = run_market_validation(
        preset=str(task["preset"]),
        seed=int(task["seed"]),
        steps=int(task["steps"]),
        config_overrides=task.get("config_overrides"),
        warmup_fraction=float(task["warmup_fraction"]),
    )
    metrics = compute_run_metrics(
        run,
        stage=str(task["stage"]),
        preset=str(task["preset"]),
        seed=int(task["seed"]),
        config_label=str(task["config_label"]),
        knob_name=task.get("knob_name"),
        knob_scale=task.get("knob_scale"),
        repeat_idx=task.get("repeat_idx"),
    )
    failures = collect_invariant_failures(
        run,
        stage=str(task["stage"]),
        preset=str(task["preset"]),
        seed=int(task["seed"]),
    )
    return metrics, failures.to_dict(orient="records")


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
            _failure_row(
                preset=preset,
                seed=seed,
                stage=stage,
                step=np.nan,
                event_idx=np.nan,
                invariant_name="run_failed",
                details=run.error_message or "run failed",
            )
        )

    if not _event_order_ok(run.event_history):
        failures.extend(
            _event_order_failures(
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
                _failure_row(
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
                _failure_row(
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
                _failure_row(
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
        _debug_alignment_failures(
            run.debug_history,
            run.event_history,
            preset=preset,
            seed=seed,
            stage=stage,
        )
    )

    for label, snapshot in (("start_snapshot", run.start_snapshot), ("end_snapshot", run.end_snapshot)):
        failures.extend(_snapshot_failures(snapshot, preset=preset, seed=seed, stage=stage, label=label))

    visual_rows = getattr(run.market, "_visual_history", []) if run.market is not None else []
    for index, row in enumerate(visual_rows):
        bid_values = np.asarray(row.bid_qty, dtype=float)
        ask_values = np.asarray(row.ask_qty, dtype=float)
        if np.any(bid_values[np.isfinite(bid_values)] < 0.0):
            failures.append(
                _failure_row(
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
                _failure_row(
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


def _analysis_history(run: ValidationRun) -> pd.DataFrame:
    if run.history.empty:
        return run.history
    start_step = _analysis_start_step(run)
    history = run.history.loc[run.history["step"] >= start_step].copy()
    return history if not history.empty else run.history.copy()


def _analysis_events(run: ValidationRun) -> pd.DataFrame:
    if run.event_history.empty:
        return run.event_history
    start_step = _analysis_start_step(run)
    events = run.event_history.loc[run.event_history["step"] >= start_step].copy()
    return events if not events.empty else run.event_history.copy()


def _analysis_debug(run: ValidationRun) -> pd.DataFrame:
    if run.debug_history.empty:
        return run.debug_history
    start_step = _analysis_start_step(run)
    debug = run.debug_history.loc[run.debug_history["step"] >= start_step].copy()
    return debug if not debug.empty else run.debug_history.copy()


def _analysis_start_step(run: ValidationRun) -> int:
    if run.requested_steps <= 0:
        return 0
    return int(min(max(run.warmup_cutoff_step, 1), run.requested_steps))


def _analysis_step_index(run: ValidationRun) -> np.ndarray:
    if run.requested_steps <= 0:
        return np.array([0], dtype=int)
    start_step = _analysis_start_step(run)
    if start_step > run.requested_steps:
        start_step = run.requested_steps
    return np.arange(start_step, run.requested_steps + 1, dtype=int)


def _phase_metrics(history: pd.DataFrame, events: pd.DataFrame, market_events: pd.DataFrame) -> dict[str, float]:
    rows: dict[str, float] = {}
    spreads: list[float] = []
    fills: list[float] = []
    for phase in PHASE_ORDER:
        phase_history = history.loc[history["session_phase"] == phase]
        phase_events = events.loc[events["session_phase"] == phase]
        phase_market_events = market_events.loc[market_events["session_phase"] == phase]
        phase_steps = max(len(phase_history), 1)
        mean_spread = _safe_mean(phase_history["spread"]) if not phase_history.empty else 0.0
        total_fill_qty = float(phase_events["fill_qty"].sum()) if not phase_events.empty else 0.0
        rows[f"phase_{phase}_mean_spread"] = mean_spread
        rows[f"phase_{phase}_market_order_intensity"] = float(len(phase_market_events) / phase_steps)
        rows[f"phase_{phase}_total_fill_qty"] = total_fill_qty
        rows[f"phase_{phase}_realized_vol"] = _safe_mean(phase_history["realized_vol"]) if not phase_history.empty else 0.0
        spreads.append(mean_spread)
        fills.append(total_fill_qty)
    rows["phase_spread_range"] = float(max(spreads) - min(spreads)) if spreads else 0.0
    rows["phase_fill_range"] = float(max(fills) - min(fills)) if fills else 0.0
    return rows


def _shock_vs_calm_abs_return(history: pd.DataFrame, debug: pd.DataFrame) -> tuple[float, float]:
    if history.empty:
        return 0.0, 0.0
    if debug.empty:
        calm = history["mid_price"].diff().abs().fillna(0.0)
        return 0.0, _safe_mean(calm)

    per_step_shock = (
        debug.groupby("step")["shock_state"]
        .agg(lambda states: "none" if (states == "none").all() else next(value for value in states if value != "none"))
        .rename("shock_state")
    )
    joined = history.set_index("step").join(per_step_shock, how="left").fillna({"shock_state": "none"})
    abs_return = joined["mid_price"].diff().abs().fillna(0.0)
    shock_abs = abs_return.loc[joined["shock_state"] != "none"]
    calm_abs = abs_return.loc[joined["shock_state"] == "none"]
    return _safe_mean(shock_abs), _safe_mean(calm_abs)


def _meta_directional_ratios(joined: pd.DataFrame) -> tuple[float, float]:
    if joined.empty:
        return 0.0, 0.0
    market_joined = joined.loc[joined["event_type"] == "market"].copy()
    if market_joined.empty:
        return 0.0, 0.0
    signs = market_joined["side"].map({"buy": 1.0, "sell": -1.0}).astype(float)
    active = signs.loc[market_joined["meta_order_id"].notna()]
    inactive = signs.loc[market_joined["meta_order_id"].isna()]
    return _directional_ratio(active), _directional_ratio(inactive)


def _per_step_event_count(step_index: np.ndarray, events: pd.DataFrame) -> pd.Series:
    if len(step_index) == 0:
        return pd.Series(dtype=float)
    counts = events.groupby("step")["event_type"].count() if not events.empty else pd.Series(dtype=float)
    return counts.reindex(step_index, fill_value=0.0).astype(float)


def _join_events_and_debug(events: pd.DataFrame, debug: pd.DataFrame) -> pd.DataFrame:
    if events.empty or debug.empty:
        return pd.DataFrame()
    return events.merge(debug, on=["step", "event_idx"], how="inner", suffixes=("_event", "_debug"))


def _event_order_ok(events: pd.DataFrame) -> bool:
    if len(events) < 2:
        return True
    steps = events["step"].to_numpy(dtype=int)
    event_idx = events["event_idx"].to_numpy(dtype=int)
    return bool(np.all((steps[1:] > steps[:-1]) | ((steps[1:] == steps[:-1]) & (event_idx[1:] > event_idx[:-1]))))


def _event_order_failures(
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
            _failure_row(
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


def _debug_alignment_failures(
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
            _failure_row(
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
            _failure_row(
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
                _failure_row(
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


def _snapshot_failures(
    snapshot: dict[str, Any] | None,
    *,
    preset: str,
    seed: int,
    stage: str,
    label: str,
) -> list[dict[str, Any]]:
    if snapshot is None:
        return [
            _failure_row(
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
            _failure_row(
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
            _failure_row(
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
            _failure_row(
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
            _failure_row(
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
            _failure_row(
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


def _table_nonfinite_failures(frame: pd.DataFrame, *, stage: str) -> pd.DataFrame:
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
            _failure_row(
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


def _reproducibility_failures(reproducibility: pd.DataFrame) -> pd.DataFrame:
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
                _failure_row(
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


def _select_diagnostics_seeds(
    baseline_metrics: pd.DataFrame,
    *,
    presets: Sequence[str],
    policy: str,
) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for preset in presets:
        frame = baseline_metrics.loc[baseline_metrics["preset"] == preset].copy()
        if frame.empty:
            continue
        if policy == "median-realized-vol":
            target = float(frame["realized_vol"].median())
            frame = frame.assign(_distance=(frame["realized_vol"] - target).abs())
            chosen = frame.sort_values(["_distance", "seed"]).iloc[0]
            seeds[preset] = int(chosen["seed"])
        else:
            seeds[preset] = int(frame.sort_values("seed").iloc[0]["seed"])
    return seeds


def _evaluate_stylized_facts(baseline_summary: pd.DataFrame) -> dict[str, bool]:
    if baseline_summary.empty:
        return {
            "abs_return_clustering": False,
            "spread_variation": False,
            "imbalance_signal": False,
            "phase_separation": False,
            "event_clustering": False,
            "shock_response": False,
            "meta_directionality": False,
        }

    return {
        "abs_return_clustering": bool((baseline_summary["abs_return_acf1_mean"] > 0.0).all()),
        "spread_variation": bool((baseline_summary["spread_unique_count_mean"] >= 3.0).all()),
        "imbalance_signal": bool((baseline_summary["imbalance_next_mid_return_corr_mean"] > 0.0).all()),
        "phase_separation": bool(
            (baseline_summary["phase_spread_range_mean"] > 0.001).any()
            or (baseline_summary["phase_fill_range_mean"] > 1.0).any()
        ),
        "event_clustering": bool(
            (
                (baseline_summary["buy_event_count_acf1_mean"] > 0.0)
                & (baseline_summary["cancel_event_count_acf1_mean"] > 0.0)
            ).sum()
            >= 2
        ),
        "shock_response": bool((baseline_summary["shock_to_calm_ratio_mean"] > 1.0).sum() >= 2),
        "meta_directionality": bool(
            (baseline_summary["meta_active_directional_ratio_mean"] > baseline_summary["meta_inactive_directional_ratio_mean"]).sum()
            >= 2
        ),
    }


def _evaluate_sensitivity_summary(sensitivity_summary: pd.DataFrame) -> dict[str, bool]:
    if sensitivity_summary.empty:
        return {}
    checks: dict[str, bool] = {}
    for knob_name, frame in sensitivity_summary.groupby("knob_name", sort=False):
        checks[knob_name] = bool(frame["direction_ok"].iloc[0])
    return checks


def _evaluate_seed_stability(baseline_metrics: pd.DataFrame, baseline_summary: pd.DataFrame) -> dict[str, Any]:
    if baseline_metrics.empty or baseline_summary.empty:
        return {"passes": False, "details": {}, "qualitative_conclusions_hold": False}

    core_metrics = (
        "mean_spread",
        "realized_vol",
        "events_per_step",
        "imbalance_next_mid_return_corr",
        "abs_return_acf1",
    )
    details: dict[str, Any] = {}
    passes = True
    for preset, frame in baseline_metrics.groupby("preset", sort=False):
        row: dict[str, float] = {}
        high_cv_count = 0
        for metric in core_metrics:
            if metric not in frame.columns:
                row[f"{metric}_cv"] = 0.0
                continue
            mean_value = _safe_mean(frame[metric])
            std_value = _safe_std(frame[metric])
            cv_value = _coefficient_of_variation(mean_value, std_value)
            row[f"{metric}_cv"] = cv_value
            if cv_value > 0.5:
                high_cv_count += 1
        row["high_cv_count"] = float(high_cv_count)
        details[preset] = row
        if high_cv_count > 2:
            passes = False

    trimmed = _remove_worst_seed_per_preset(baseline_metrics, core_metrics)
    trimmed_summary = summarize_validation_grid(trimmed)
    full_index = baseline_summary
    trimmed_index = trimmed_summary.set_index("preset")
    qualitative_conclusions_hold = bool(
        _preset_compare(full_index, "volatile", "balanced", "realized_vol_mean")
        == _preset_compare(trimmed_index, "volatile", "balanced", "realized_vol_mean")
        and _preset_compare(full_index, "volatile", "balanced", "mean_spread_mean", comparison="ge")
        == _preset_compare(trimmed_index, "volatile", "balanced", "mean_spread_mean", comparison="ge")
        and _preset_compare(full_index, "trend", "balanced", "trade_sign_acf1_mean")
        == _preset_compare(trimmed_index, "trend", "balanced", "trade_sign_acf1_mean")
    )
    return {
        "passes": bool(passes and qualitative_conclusions_hold),
        "details": details,
        "qualitative_conclusions_hold": qualitative_conclusions_hold,
    }


def _remove_worst_seed_per_preset(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    keep_rows = []
    for _, group in frame.groupby("preset", sort=False):
        if len(group) <= 1:
            keep_rows.append(group.copy())
            continue
        zscores = np.zeros(len(group), dtype=float)
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=float)
            std = float(np.std(values))
            if std <= EPSILON:
                continue
            zscores += np.abs((values - float(np.mean(values))) / std)
        drop_position = int(np.argmax(zscores))
        keep_rows.append(group.drop(group.index[drop_position]))
    return pd.concat(keep_rows, ignore_index=True, sort=False)


def _leave_one_out_centroid_accuracy(frame: pd.DataFrame, *, features: Sequence[str]) -> float:
    if frame.empty or frame["preset"].nunique() < 2:
        return 0.0
    data = frame[list(features)].to_numpy(dtype=float)
    labels = frame["preset"].to_numpy()
    correct = 0
    total = 0
    for index in range(len(frame)):
        train_mask = np.ones(len(frame), dtype=bool)
        train_mask[index] = False
        train = data[train_mask]
        train_labels = labels[train_mask]
        if len(np.unique(train_labels)) < 2:
            continue
        mean = np.mean(train, axis=0)
        std = np.std(train, axis=0)
        std[std <= EPSILON] = 1.0
        train_scaled = (train - mean) / std
        test_scaled = (data[index] - mean) / std
        centroids = {
            label: np.mean(train_scaled[train_labels == label], axis=0)
            for label in np.unique(train_labels)
        }
        predicted = min(centroids.items(), key=lambda item: float(np.linalg.norm(test_scaled - item[1])))[0]
        correct += int(predicted == labels[index])
        total += 1
    return float(correct / max(total, 1))


def _write_validation_summary(
    *,
    outpath: Path,
    presets: Sequence[str],
    baseline_seed_list: Sequence[int],
    sensitivity_seed_list: Sequence[int],
    soak_seed_list: Sequence[int],
    baseline_steps: int,
    sensitivity_steps: int,
    long_run_steps: int,
    warmup_fraction: float,
    run_metrics: pd.DataFrame,
    preset_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    invariant_failures: pd.DataFrame,
    acceptance: Mapping[str, Any],
    diagnostics_paths: Mapping[str, Path],
) -> None:
    baseline_summary = preset_summary.loc[preset_summary["stage"] == "baseline"]
    soak_summary = preset_summary.loc[preset_summary["stage"] == "soak"]
    lines = [
        "# Orderwave 최종 검증 리포트",
        "",
        "## 1. 실험 설정",
        f"- presets: {', '.join(presets)}",
        f"- baseline: {len(baseline_seed_list)} seeds x {baseline_steps:,} steps",
        f"- sensitivity: {len(sensitivity_seed_list)} seeds x {sensitivity_steps:,} steps x {len(DEFAULT_SENSITIVITY_SCALES)} scales",
        f"- long-run soak: {len(soak_seed_list)} seeds x {long_run_steps:,} steps",
        f"- warm-up fraction: {warmup_fraction:.2f}",
        "",
        "## 2. 하드 게이트",
        f"- invariants: `{_pass_fail(acceptance['invariants_ok'])}`",
        f"- reproducibility: `{_pass_fail(acceptance['reproducibility_ok'])}`",
        f"- performance: `{_pass_fail(acceptance['performance_ok'])}`",
        "",
        "### 성능 체크",
        _markdown_bullets(acceptance["performance_checks"]),
        "",
        "## 3. baseline preset summary",
        _markdown_table(
            baseline_summary[
                [
                    "preset",
                    "runs",
                    "mean_spread_mean",
                    "realized_vol_mean",
                    "trade_sign_acf1_mean",
                    "events_per_step_mean",
                    "steps_per_second_mean",
                    "run_failures",
                ]
            ].round(4)
        ) if not baseline_summary.empty else "_no data_",
        "",
        "## 4. reproducibility",
        _markdown_table(reproducibility) if not reproducibility.empty else "_no data_",
        "",
        "## 5. sensitivity summary",
        _markdown_table(
            sensitivity_summary[
                [
                    "knob_name",
                    "knob_scale",
                    "target_metric",
                    "direction_ok",
                ]
            ]
        ) if not sensitivity_summary.empty else "_no data_",
        "",
        "## 6. long-run soak summary",
        _markdown_table(
            soak_summary[
                [
                    "preset",
                    "runs",
                    "steps_per_second_mean",
                    "peak_memory_mb_mean",
                    "bytes_per_logged_event_mean",
                    "run_failures",
                    "memory_growth_failures",
                ]
            ].round(4)
        ) if not soak_summary.empty else "_no data_",
        "",
        "## 7. soft gates",
        f"- preset separation: `{_pass_fail(acceptance['preset_separation_ok'])}`",
        f"- stylized facts / time structure: `{_pass_fail(acceptance['stylized_facts_ok'])}`",
        f"- sensitivity: `{_pass_fail(acceptance['sensitivity_ok'])}`",
        f"- seed stability: `{_pass_fail(acceptance['seed_stability_ok'])}`",
        "",
        "### preset separation details",
        _markdown_bullets(acceptance["preset_checks"]),
        "",
        "### stylized facts",
        _markdown_bullets(acceptance["stylized_checks"]),
        "",
        "### sensitivity direction checks",
        _markdown_bullets(acceptance["sensitivity_checks"]),
        "",
        "## 8. invariant failures",
        f"- failure rows: `{len(invariant_failures)}`",
        _markdown_table(invariant_failures.head(20)) if not invariant_failures.empty else "_none_",
        "",
        "## 9. diagnostics images",
    ]
    for preset in presets:
        path = diagnostics_paths.get(preset)
        if path is None:
            lines.append(f"- {preset}: `_not generated_`")
        else:
            lines.append(f"- {preset}: `{path.name}`")
    lines.extend(
        [
            "",
            "## 10. 최종 판정",
            f"- 판정: `{acceptance['decision']}`",
            f"- 평가 관점: synthetic market-state generator",
            f"- 핵심 강점: {', '.join(acceptance['strengths'])}",
            f"- 핵심 약점: {', '.join(acceptance['weaknesses']) if acceptance['weaknesses'] else '없음'}",
            f"- 즉시 채택 가능 범위: {acceptance['immediate_scope']}",
            f"- 채택 전 보완 필요 항목: {acceptance['required_fix']}",
            "",
            acceptance["conclusion_market_state"],
            acceptance["conclusion_scope"],
            acceptance["conclusion_required_fix"],
            "",
        ]
    )
    outpath.write_text("\n".join(lines), encoding="utf-8")


def _write_acceptance_decision(*, outpath: Path, acceptance: Mapping[str, Any]) -> None:
    lines = [
        "# Acceptance Decision",
        "",
        f"- final verdict: `{acceptance['decision']}`",
        f"- invariants: `{_pass_fail(acceptance['invariants_ok'])}`",
        f"- reproducibility: `{_pass_fail(acceptance['reproducibility_ok'])}`",
        f"- preset separation: `{_pass_fail(acceptance['preset_separation_ok'])}`",
        f"- stylized facts: `{_pass_fail(acceptance['stylized_facts_ok'])}`",
        f"- sensitivity: `{_pass_fail(acceptance['sensitivity_ok'])}`",
        f"- seed stability: `{_pass_fail(acceptance['seed_stability_ok'])}`",
        f"- performance: `{_pass_fail(acceptance['performance_ok'])}`",
        "",
        acceptance["conclusion_market_state"],
        acceptance["conclusion_scope"],
        acceptance["conclusion_required_fix"],
        "",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def _failure_row(
    *,
    preset: str,
    seed: int,
    stage: str,
    step: float,
    event_idx: float,
    invariant_name: str,
    details: str,
) -> dict[str, Any]:
    return {
        "preset": preset,
        "seed": int(seed),
        "stage": stage,
        "step": step,
        "event_idx": event_idx,
        "invariant_name": invariant_name,
        "details": details,
    }


def _concat_failures(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    return pd.concat(valid, ignore_index=True, sort=False).reindex(columns=INVARIANT_FAILURE_COLUMNS)


def _tick_size_from_run(run: ValidationRun) -> float:
    if run.market is not None:
        return float(run.market.tick_size)
    if not run.history.empty and len(run.history) > 1:
        spread = run.history["spread"].replace(0.0, np.nan).dropna()
        if not spread.empty:
            return float(spread.min())
    return 0.01


def _stable_frame_hash(frame: pd.DataFrame) -> str:
    payload = {
        "columns": list(frame.columns),
        "records": [_normalize_value(record) for record in frame.to_dict(orient="records")],
    }
    return _stable_object_hash(payload)


def _deterministic_metric_view(metrics: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "steps_per_second",
        "events_logged_per_second",
        "peak_memory_mb",
        "bytes_per_logged_event",
        "memory_metric",
        "memory_growth_without_recovery",
        "error_message",
    }
    return {key: value for key, value in metrics.items() if key not in excluded}


def _stable_object_hash(value: Any) -> str:
    serialized = json.dumps(_normalize_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, np.generic):
        return _normalize_value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if pd.isna(value):
        return "NaN"
    return value


def _regime_switch_rate(history: pd.DataFrame) -> float:
    if len(history) < 2:
        return 0.0
    switches = history["regime"].ne(history["regime"].shift()).iloc[1:]
    return float(switches.mean())


def _sensitivity_target_metric(knob_name: str) -> str:
    mapping = {
        "limit_rate_scale": "limit_share_mean",
        "market_rate_scale": "events_per_step_market_mean",
        "cancel_rate_scale": "cancel_share_mean",
        "fair_price_vol_scale": "realized_vol_mean",
        "regime_transition_scale": "regime_switch_rate_mean",
        "seasonality_scale": "phase_structure_signal_mean",
        "excitation_scale": "event_clustering_mean",
        "meta_order_scale": "meta_active_directional_ratio_mean",
        "shock_scale": "shock_to_calm_ratio_mean",
    }
    return mapping[knob_name]


def _quasi_monotonic(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return False
    diffs = np.diff(finite)
    signs = np.sign(diffs[np.abs(diffs) > EPSILON])
    if len(signs) <= 1:
        return True
    reversals = int(np.sum(signs[1:] < signs[:-1]))
    return reversals <= 1


def _preset_compare(frame: pd.DataFrame, left: str, right: str, column: str, *, comparison: str = "gt") -> bool:
    if left not in frame.index or right not in frame.index:
        return False
    left_value = float(frame.loc[left, column])
    right_value = float(frame.loc[right, column])
    if comparison == "ge":
        return bool(left_value >= right_value)
    return bool(left_value > right_value)


def _directional_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    values = values.astype(float)
    return float(abs(values.sum()) / max(len(values), 1))


def _safe_mean(values: pd.Series | Sequence[float]) -> float:
    series = pd.Series(values, dtype=float)
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0
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
    return 0.0 if np.isnan(corr) else corr


def _safe_autocorr(values: pd.Series | Sequence[float], lag: int) -> float:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) <= lag:
        return 0.0
    corr = float(series.autocorr(lag=lag))
    return 0.0 if np.isnan(corr) else corr


def _coefficient_of_variation(mean_value: float, std_value: float) -> float:
    if not np.isfinite(mean_value) or abs(mean_value) <= EPSILON:
        return float("inf") if std_value > 0.0 else 0.0
    return float(abs(std_value) / abs(mean_value))


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_no data_"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown_bullets(values: Mapping[str, Any]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- `{key}`: {value}" for key, value in values.items())


def _pass_fail(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"
