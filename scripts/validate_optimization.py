from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from orderwave.validation import (
    BASELINE_THROUGHPUT_FLOOR,
    benchmark_logging_modes,
    compute_run_metrics,
    run_market_validation,
)


def _seed_metrics(*, preset: str, seeds: list[int], steps: int, warmup_fraction: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        run = run_market_validation(
            preset=preset,
            seed=seed,
            steps=steps,
            warmup_fraction=warmup_fraction,
        )
        metrics = compute_run_metrics(
            run,
            stage="optimization",
            preset=preset,
            seed=seed,
            config_label="optimization",
        )
        rows.append(
            {
                "seed": int(seed),
                "steps": int(steps),
                "steps_per_second": float(metrics["steps_per_second"]),
                "events_per_step": float(metrics["events_per_step"]),
                "peak_memory_mb": float(metrics["peak_memory_mb"]),
                "bytes_per_logged_event": float(metrics["bytes_per_logged_event"]),
                "run_failed": bool(metrics["run_failed"]),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _summary_lines(
    *,
    preset: str,
    floor: float,
    metrics: pd.DataFrame,
    logging_compare: pd.DataFrame,
) -> list[str]:
    full_compare = logging_compare.loc[logging_compare["logging_mode"] == "full"].iloc[0]
    compact_compare = logging_compare.loc[logging_compare["logging_mode"] == "history_only"].iloc[0]
    mean_speed = float(metrics["steps_per_second"].mean())
    min_speed = float(metrics["steps_per_second"].min())
    max_speed = float(metrics["steps_per_second"].max())
    failed_runs = int(metrics["run_failed"].sum())
    pass_floor = bool(failed_runs == 0 and mean_speed >= floor)

    return [
        f"# orderwave optimization validation",
        "",
        f"- preset: `{preset}`",
        f"- throughput floor: `{floor:.0f} steps/s`",
        f"- mean throughput: `{mean_speed:.3f} steps/s`",
        f"- min throughput: `{min_speed:.3f} steps/s`",
        f"- max throughput: `{max_speed:.3f} steps/s`",
        f"- mean events/step: `{float(metrics['events_per_step'].mean()):.3f}`",
        f"- mean peak memory: `{float(metrics['peak_memory_mb'].mean()):.3f} MB`",
        f"- failed runs: `{failed_runs}`",
        f"- floor check: `{'PASS' if pass_floor else 'FAIL'}`",
        "",
        "## Logging mode comparison",
        f"- comparison seed: `{int(full_compare['seed'])}`",
        f"- full throughput: `{float(full_compare['steps_per_second']):.3f} steps/s`",
        f"- history_only throughput: `{float(compact_compare['steps_per_second']):.3f} steps/s`",
        f"- history_only throughput improvement: `{float(compact_compare['throughput_improvement_pct_vs_full']):.3f}%`",
        f"- full peak memory increase: `{float(full_compare['peak_memory_increase_mb']):.3f} MB`",
        f"- history_only peak memory increase: `{float(compact_compare['peak_memory_increase_mb']):.3f} MB`",
        f"- history_only memory reduction: `{float(compact_compare['peak_memory_reduction_pct_vs_full']):.3f}%`",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a targeted optimization validation for orderwave.")
    parser.add_argument("--preset", type=str, default="balanced")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--floor", type=float, default=None)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts") / "optimization-validation",
    )
    args = parser.parse_args()

    seeds = list(range(1, int(args.seeds) + 1))
    floor = float(args.floor) if args.floor is not None else float(BASELINE_THROUGHPUT_FLOOR.get(args.preset, 0.0))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = _seed_metrics(
        preset=args.preset,
        seeds=seeds,
        steps=int(args.steps),
        warmup_fraction=float(args.warmup_fraction),
    )
    logging_compare = benchmark_logging_modes(
        preset=args.preset,
        seed=seeds[0],
        steps=int(args.steps),
        warmup_fraction=float(args.warmup_fraction),
    )

    metrics_path = outdir / "optimization_metrics.csv"
    logging_path = outdir / "optimization_logging_modes.csv"
    summary_path = outdir / "optimization_summary.md"

    metrics.to_csv(metrics_path, index=False)
    logging_compare.to_csv(logging_path, index=False)
    summary_path.write_text(
        "\n".join(
            _summary_lines(
                preset=args.preset,
                floor=floor,
                metrics=metrics,
                logging_compare=logging_compare,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"preset={args.preset}")
    print(f"steps={int(args.steps)}")
    print(f"seeds={len(seeds)}")
    print(f"mean_steps_per_second={float(metrics['steps_per_second'].mean()):.3f}")
    print(f"throughput_floor={floor:.3f}")
    print(f"failed_runs={int(metrics['run_failed'].sum())}")
    print(f"metrics_csv={metrics_path}")
    print(f"logging_csv={logging_path}")
    print(f"summary_md={summary_path}")


if __name__ == "__main__":
    main()
