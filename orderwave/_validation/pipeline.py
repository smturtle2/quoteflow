from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .decision import evaluate_validation_results
from .metrics import (
    reproducibility_failures,
    summarize_sensitivity_grid,
    summarize_validation_grid,
    table_nonfinite_failures,
)
from .reporting import write_acceptance_decision, write_validation_summary
from .reproducibility import run_reproducibility_checks
from .shared import (
    DEFAULT_PRESETS,
    DEFAULT_SENSITIVITY_KNOBS,
    DEFAULT_SENSITIVITY_SCALES,
    INVARIANT_FAILURE_COLUMNS,
    ValidationPipelineResult,
)
from .single_run import run_market_validation
from .tasks import execute_run_grid, execute_sensitivity_grid, select_diagnostics_seeds


def run_validation_pipeline(
    *,
    outdir: Path,
    profile_name: str = "quality_regression",
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

    baseline_metrics, baseline_failures = execute_run_grid(
        stage="baseline",
        presets=presets,
        seeds=baseline_seed_list,
        steps=baseline_steps,
        warmup_fraction=warmup_fraction,
        jobs=jobs,
    )
    sensitivity_metrics, sensitivity_failures = execute_sensitivity_grid(
        preset="balanced",
        seeds=sensitivity_seed_list,
        steps=sensitivity_steps,
        warmup_fraction=warmup_fraction,
        knobs=sensitivity_knobs,
        scales=sensitivity_scales,
        jobs=jobs,
    )
    soak_metrics, soak_failures = execute_run_grid(
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
        profile_name=profile_name,
    )

    diagnostics_paths: dict[str, Path] = {}
    if render_diagnostics:
        diagnostics_seeds = select_diagnostics_seeds(
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
        profile_name=profile_name,
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        sensitivity_summary=sensitivity_summary,
        invariant_failures=invariant_failures,
        reproducibility=reproducibility,
        acceptance=acceptance,
        diagnostics_paths=diagnostics_paths,
        artifact_paths=artifact_paths,
    )
