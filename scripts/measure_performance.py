from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from orderwave.validation import BASELINE_THROUGHPUT_FLOOR, measure_performance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure orderwave simulator throughput and memory.")
    parser.add_argument("--preset", type=str, default="balanced")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=1, help="First seed value in the sweep.")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--warmup-fraction", type=float, default=0.10)
    parser.add_argument("--floor", type=float, default=None, help="Optional throughput floor override.")
    parser.add_argument(
        "--logging-compare-seed",
        type=int,
        default=None,
        help="Seed used for the full vs history_only logging comparison.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts" / "performance",
        help="Directory for CSV and markdown artifacts.",
    )
    return parser.parse_args()


def _summary_lines(*, summary_row: dict[str, object], logging_compare_seed: int, logging_compare_row_full: dict[str, object], logging_compare_row_compact: dict[str, object]) -> list[str]:
    return [
        "# orderwave performance report",
        "",
        f"- preset: `{summary_row['preset']}`",
        f"- seeds: `{summary_row['seeds']}`",
        f"- steps per seed: `{summary_row['steps']}`",
        f"- throughput floor: `{float(summary_row['throughput_floor']):.0f} steps/s`",
        f"- mean throughput: `{float(summary_row['mean_steps_per_second']):.3f} steps/s`",
        f"- min throughput: `{float(summary_row['min_steps_per_second']):.3f} steps/s`",
        f"- max throughput: `{float(summary_row['max_steps_per_second']):.3f} steps/s`",
        f"- mean events/step: `{float(summary_row['mean_events_per_step']):.3f}`",
        f"- mean peak memory: `{float(summary_row['mean_peak_memory_mb']):.3f} MB`",
        f"- mean bytes/logged event: `{float(summary_row['mean_bytes_per_logged_event']):.3f}`",
        f"- failed runs: `{int(summary_row['failed_runs'])}`",
        f"- floor check: `{'PASS' if bool(summary_row['floor_pass']) else 'FAIL'}`",
        "",
        "## Logging mode comparison",
        f"- comparison seed: `{logging_compare_seed}`",
        f"- full throughput: `{float(logging_compare_row_full['steps_per_second']):.3f} steps/s`",
        f"- history_only throughput: `{float(logging_compare_row_compact['steps_per_second']):.3f} steps/s`",
        f"- history_only throughput improvement: `{float(logging_compare_row_compact['throughput_improvement_pct_vs_full']):.3f}%`",
        f"- full peak memory increase: `{float(logging_compare_row_full['peak_memory_increase_mb']):.3f} MB`",
        f"- history_only peak memory increase: `{float(logging_compare_row_compact['peak_memory_increase_mb']):.3f} MB`",
        f"- history_only memory reduction: `{float(logging_compare_row_compact['peak_memory_reduction_pct_vs_full']):.3f}%`",
    ]


def main() -> None:
    args = parse_args()
    seed_values = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    floor = float(args.floor) if args.floor is not None else float(BASELINE_THROUGHPUT_FLOOR.get(args.preset, 0.0))

    result = measure_performance(
        preset=args.preset,
        seeds=seed_values,
        steps=int(args.steps),
        warmup_fraction=float(args.warmup_fraction),
        throughput_floor=floor,
        logging_compare_seed=args.logging_compare_seed,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.outdir / "performance_metrics.csv"
    summary_path = args.outdir / "performance_summary.csv"
    logging_path = args.outdir / "performance_logging_modes.csv"
    report_path = args.outdir / "performance_summary.md"

    result["seed_metrics"].to_csv(metrics_path, index=False)
    result["summary"].to_csv(summary_path, index=False)
    result["logging_compare"].to_csv(logging_path, index=False)

    summary_row = result["summary"].iloc[0].to_dict()
    full_row = result["logging_compare"].loc[result["logging_compare"]["logging_mode"] == "full"].iloc[0].to_dict()
    compact_row = result["logging_compare"].loc[result["logging_compare"]["logging_mode"] == "history_only"].iloc[0].to_dict()
    compare_seed = int(args.logging_compare_seed) if args.logging_compare_seed is not None else seed_values[0]
    report_path.write_text(
        "\n".join(
            _summary_lines(
                summary_row=summary_row,
                logging_compare_seed=compare_seed,
                logging_compare_row_full=full_row,
                logging_compare_row_compact=compact_row,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"preset={args.preset}")
    print(f"steps={int(args.steps)}")
    print(f"seeds={len(seed_values)}")
    print(f"mean_steps_per_second={float(summary_row['mean_steps_per_second']):.3f}")
    print(f"throughput_floor={floor:.3f}")
    print(f"floor_pass={bool(summary_row['floor_pass'])}")
    print(f"metrics_csv={metrics_path}")
    print(f"summary_csv={summary_path}")
    print(f"logging_csv={logging_path}")
    print(f"summary_md={report_path}")


if __name__ == "__main__":
    main()
