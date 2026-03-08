from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import matplotlib

matplotlib.use("Agg")

from orderwave.validation import (
    DEFAULT_PRESETS,
    compare_validation_baseline,
    load_validation_baseline,
    run_validation_pipeline,
    write_validation_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final orderwave validation pipeline.")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(DEFAULT_PRESETS),
        help="Preset names to validate.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts" / "validation",
        help="Output directory for CSV, markdown, and diagnostics artifacts.",
    )
    parser.add_argument("--seed-start", type=int, default=1, help="First seed used in baseline sweeps.")
    parser.add_argument("--warmup-fraction", type=float, default=0.10, help="Warm-up fraction excluded from statistics.")
    parser.add_argument("--baseline-seeds", type=int, default=20, help="Seed count per preset for baseline runs.")
    parser.add_argument("--baseline-steps", type=int, default=20_000, help="Steps per preset/seed baseline run.")
    parser.add_argument("--sensitivity-seeds", type=int, default=8, help="Seed count for one-at-a-time sensitivity sweeps.")
    parser.add_argument("--sensitivity-steps", type=int, default=15_000, help="Steps per sensitivity run.")
    parser.add_argument("--long-run-seeds", type=int, default=3, help="Seed count per preset for soak runs.")
    parser.add_argument("--long-run-steps", type=int, default=200_000, help="Steps per preset/seed soak run.")
    parser.add_argument(
        "--diagnostics-seed-policy",
        choices=("median-realized-vol", "first-seed"),
        default="median-realized-vol",
        help="How representative diagnostics seeds are chosen per preset.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Worker process count for baseline, sensitivity, and soak sweeps.")
    parser.add_argument("--baseline-json", type=Path, default=None, help="Golden validation baseline JSON to compare against.")
    parser.add_argument("--write-baseline-json", type=Path, default=None, help="Write a golden validation baseline JSON from this run.")
    parser.add_argument(
        "--fail-on-baseline-drift",
        action="store_true",
        help="Exit non-zero when the validation run drifts from the provided golden baseline.",
    )

    # Backward-compatible aliases used by older docs and notes.
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_steps = args.steps if args.steps is not None else args.baseline_steps
    baseline_seeds = args.seeds if args.seeds is not None else args.baseline_seeds

    result = run_validation_pipeline(
        outdir=args.outdir,
        presets=args.presets,
        baseline_seeds=baseline_seeds,
        baseline_steps=baseline_steps,
        sensitivity_seeds=args.sensitivity_seeds,
        sensitivity_steps=args.sensitivity_steps,
        long_run_seeds=args.long_run_seeds,
        long_run_steps=args.long_run_steps,
        seed_start=args.seed_start,
        warmup_fraction=args.warmup_fraction,
        diagnostics_seed_policy=args.diagnostics_seed_policy,
        jobs=max(1, int(args.jobs)),
    )

    print(f"[validation] decision={result.acceptance['decision']}")
    for name, path in result.artifact_paths.items():
        print(f"[validation] {name}={path}")
    for preset, path in result.diagnostics_paths.items():
        print(f"[validation] diagnostics_{preset}={path}")

    if args.write_baseline_json is not None:
        write_validation_baseline(args.write_baseline_json, result)
        print(f"[validation] baseline_written={args.write_baseline_json}")

    if args.baseline_json is not None:
        baseline = load_validation_baseline(args.baseline_json)
        comparison = compare_validation_baseline(result, baseline)
        print(f"[validation] baseline_match={comparison['matches']}")
        if comparison["failures"]:
            print(f"[validation] baseline_failures={json.dumps(comparison['failures'], ensure_ascii=False)}")
        if args.fail_on_baseline_drift and not comparison["matches"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
