from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Mapping, Sequence

import pandas as pd

from .invariants import collect_invariant_failures
from .path_metrics import compute_run_metrics
from .shared import INVARIANT_FAILURE_COLUMNS, concat_failures
from .single_run import run_market_validation


def execute_run_grid(
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
    return execute_tasks(tasks, jobs=jobs)


def execute_sensitivity_grid(
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
    return execute_tasks(tasks, jobs=jobs)


def execute_tasks(tasks: Sequence[dict[str, Any]], *, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tasks:
        return pd.DataFrame(), pd.DataFrame(columns=INVARIANT_FAILURE_COLUMNS)
    if jobs <= 1:
        results = [_run_validation_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(jobs), max_tasks_per_child=1) as executor:
            results = list(executor.map(_run_validation_task, tasks))

    rows = [metrics for metrics, _ in results]
    failure_frames = [pd.DataFrame.from_records(records, columns=INVARIANT_FAILURE_COLUMNS) for _, records in results]
    metrics_frame = pd.DataFrame(rows)
    sort_columns = [column for column in ("stage", "preset", "config_label", "seed") if column in metrics_frame.columns]
    if sort_columns:
        metrics_frame = metrics_frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    failures_frame = concat_failures(failure_frames)
    if not failures_frame.empty:
        failures_frame = failures_frame.sort_values(
            ["stage", "preset", "seed", "step", "event_idx", "invariant_name"],
            kind="stable",
        ).reset_index(drop=True)
    return metrics_frame, failures_frame


def run_validation_task(task: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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


def select_diagnostics_seeds(
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


def _run_validation_task(task: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return run_validation_task(task)


__all__ = [
    "execute_run_grid",
    "execute_sensitivity_grid",
    "execute_tasks",
    "run_validation_task",
    "select_diagnostics_seeds",
]
