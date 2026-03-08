from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import pandas as pd

from orderwave.history import DEBUG_COLUMNS, EVENT_COLUMNS
from orderwave.market import Market

from .decision import evaluate_validation_results
from .metrics import (
    collect_invariant_failures,
    compute_run_metrics,
    reproducibility_failures,
    summarize_sensitivity_grid,
    summarize_validation_grid,
    table_nonfinite_failures,
)
from .reporting import write_acceptance_decision, write_validation_summary
from .shared import (
    DEFAULT_PRESETS,
    DEFAULT_SENSITIVITY_KNOBS,
    DEFAULT_SENSITIVITY_SCALES,
    INVARIANT_FAILURE_COLUMNS,
    MemoryTracker,
    ValidationPipelineResult,
    ValidationRun,
    concat_failures,
    deterministic_metric_view,
    stable_frame_hash,
    stable_object_hash,
)


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

    tracker = MemoryTracker()
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
    except Exception as exc:  # pragma: no cover - failure-path guard
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

    warmup_cutoff = int(max(0, min(steps, int(float(steps) * float(warmup_fraction)))))
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
        warmup_cutoff_step=warmup_cutoff,
        run_failed=bool(run_failed),
        error_message=error_message,
    )


def run_reproducibility_checks(
    *,
    presets: Sequence[str],
    seed: int,
    steps: int,
    warmup_fraction: float = 0.10,
) -> pd.DataFrame:
    """Run repeated same-seed checks and compare against step-by-step execution."""

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
                    "history": stable_frame_hash(run.history),
                    "events": stable_frame_hash(run.event_history),
                    "debug": stable_frame_hash(run.debug_history),
                }
            )
            metric_hashes.append(stable_object_hash(deterministic_metric_view(metrics)))

        base_hash = repeated_hashes[0]
        history_hash_equal = all(item["history"] == base_hash["history"] for item in repeated_hashes[1:])
        event_hash_equal = all(item["events"] == base_hash["events"] for item in repeated_hashes[1:])
        debug_hash_equal = all(item["debug"] == base_hash["debug"] for item in repeated_hashes[1:])
        metrics_hash_equal = all(metric_hash == metric_hashes[0] for metric_hash in metric_hashes[1:])

        gen_history = repeated_runs[0].history
        gen_events = repeated_runs[0].event_history
        gen_debug = repeated_runs[0].debug_history

        rows.append(
            {
                "preset": preset,
                "seed": int(seed),
                "steps": int(steps),
                "history_hash_equal": bool(history_hash_equal),
                "event_hash_equal": bool(event_hash_equal),
                "debug_hash_equal": bool(debug_hash_equal),
                "metrics_hash_equal": bool(metrics_hash_equal),
                "gen_vs_step_history_equal": bool(gen_history.equals(step_market.get_history())),
                "gen_vs_step_event_equal": bool(gen_events.equals(step_market.get_event_history())),
                "gen_vs_step_debug_equal": bool(gen_debug.equals(step_market.get_debug_history())),
                "all_reproducible": bool(
                    history_hash_equal
                    and event_hash_equal
                    and debug_hash_equal
                    and metrics_hash_equal
                    and gen_history.equals(step_market.get_history())
                    and gen_events.equals(step_market.get_event_history())
                    and gen_debug.equals(step_market.get_debug_history())
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
    jobs: int = 1,
) -> pd.DataFrame:
    """Run one-at-a-time sensitivity experiments for one preset."""

    metrics, _ = _execute_sensitivity_grid(
        preset=preset,
        seeds=seeds,
        steps=steps,
        warmup_fraction=warmup_fraction,
        knobs=knobs,
        scales=scales,
        jobs=jobs,
    )
    return metrics


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
    sensitivity_knobs: Sequence[str] = DEFAULT_SENSITIVITY_KNOBS,
    sensitivity_scales: Sequence[float] = DEFAULT_SENSITIVITY_SCALES,
    diagnostics_seed_policy: str = "median-realized-vol",
    render_diagnostics: bool = True,
    jobs: int = 1,
) -> ValidationPipelineResult:
    """Run the validation pipeline and write repo-standard artifacts."""

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
            reproducibility_failures(reproducibility),
        ],
        ignore_index=True,
        sort=False,
    )
    invariant_failures = pd.concat(
        [
            invariant_failures,
            table_nonfinite_failures(run_metrics, stage="run_metrics"),
            table_nonfinite_failures(preset_summary, stage="preset_summary"),
            table_nonfinite_failures(sensitivity_summary, stage="sensitivity_summary"),
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

    diagnostics_paths: dict[str, Path] = {}
    if render_diagnostics:
        diagnostics_seeds = _select_diagnostics_seeds(
            baseline_metrics,
            presets=presets,
            policy=diagnostics_seed_policy,
        )
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
            except Exception:  # pragma: no cover
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
    write_validation_summary(
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
    write_acceptance_decision(
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
