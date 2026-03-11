from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .metrics import compute_run_metrics
from .shared import BASELINE_THROUGHPUT_FLOOR, EPSILON
from .single_run import run_market_validation


def benchmark_logging_modes(
    *,
    preset: str = "balanced",
    seed: int = 42,
    steps: int = 10_000,
    warmup_fraction: float = 0.10,
) -> pd.DataFrame:
    """Compare full logging against history-only mode on one configuration."""

    rows: list[dict[str, object]] = []
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
            100.0
            * (float(history_only_row["steps_per_second"]) - float(full_row["steps_per_second"]))
            / max(float(full_row["steps_per_second"]), EPSILON)
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
    """Run the canonical performance sweep for one preset."""

    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("seeds must not be empty")

    rows: list[dict[str, object]] = []
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
