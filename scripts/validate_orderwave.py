from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from orderwave.validation import (
    DEFAULT_SENSITIVITY_KNOBS,
    PHASE_ORDER,
    compute_run_metrics,
    evaluate_validation_results,
    run_market_validation,
    run_reproducibility_checks,
    run_sensitivity_grid,
    run_validation_grid,
    summarize_sensitivity_grid,
    summarize_validation_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the orderwave validation plan.")
    parser.add_argument("--steps", type=int, default=10_000, help="Steps per main validation run.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds per preset.")
    parser.add_argument("--seed-start", type=int, default=1, help="First seed in the sweep.")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["balanced", "trend", "volatile"],
        help="Preset names to validate.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "artifacts" / "validation",
        help="Output directory for CSV, PNG, and markdown artifacts.",
    )
    parser.add_argument(
        "--sensitivity-steps",
        type=int,
        default=5_000,
        help="Steps per sensitivity run.",
    )
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=8,
        help="Number of seeds used for knob sensitivity checks.",
    )
    parser.add_argument(
        "--sensitivity-scale",
        type=float,
        default=1.5,
        help="Raised scale used for knob sensitivity runs.",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the knob sensitivity stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    sensitivity_seeds = seeds[: max(1, min(args.sensitivity_seeds, len(seeds)))]

    print(f"[validation] presets={args.presets} seeds={len(seeds)} steps={args.steps}")
    run_metrics = run_validation_grid(
        presets=args.presets,
        seeds=seeds,
        steps=args.steps,
    )
    preset_summary = summarize_validation_grid(run_metrics)
    reproducibility = run_reproducibility_checks(
        presets=args.presets,
        seed=seeds[0],
        steps=min(500, max(50, args.steps // 20)),
    )

    sensitivity_metrics = pd.DataFrame()
    sensitivity_summary = pd.DataFrame()
    if not args.skip_sensitivity:
        print(
            "[validation] sensitivity="
            f"preset=balanced seeds={len(sensitivity_seeds)} steps={args.sensitivity_steps} scale={args.sensitivity_scale}"
        )
        sensitivity_metrics = run_sensitivity_grid(
            preset="balanced",
            seeds=sensitivity_seeds,
            steps=args.sensitivity_steps,
            knobs=DEFAULT_SENSITIVITY_KNOBS,
            scales=(1.0, float(args.sensitivity_scale)),
        )
        sensitivity_summary = summarize_sensitivity_grid(sensitivity_metrics)

    verdict = evaluate_validation_results(
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        reproducibility=reproducibility,
        sensitivity_summary=sensitivity_summary if not sensitivity_summary.empty else None,
    )

    run_metrics.to_csv(args.outdir / "validation-runs.csv", index=False)
    preset_summary.to_csv(args.outdir / "validation-summary.csv", index=False)
    reproducibility.to_csv(args.outdir / "validation-reproducibility.csv", index=False)
    if not sensitivity_metrics.empty:
        sensitivity_metrics.to_csv(args.outdir / "validation-sensitivity-runs.csv", index=False)
    if not sensitivity_summary.empty:
        sensitivity_summary.to_csv(args.outdir / "validation-sensitivity-summary.csv", index=False)

    _render_preset_metrics(run_metrics, args.outdir / "preset-metrics.png")
    _render_phase_profiles(preset_summary, args.outdir / "phase-profiles.png")
    if not sensitivity_summary.empty:
        _render_sensitivity(sensitivity_summary, args.outdir / "sensitivity.png")
    _render_diagnostics_examples(
        presets=args.presets,
        seed=seeds[0],
        steps=max(args.steps, 2_000),
        outdir=args.outdir,
    )
    _write_report(
        outpath=args.outdir / "report.md",
        presets=args.presets,
        seeds=seeds,
        steps=args.steps,
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        reproducibility=reproducibility,
        sensitivity_summary=sensitivity_summary,
        verdict=verdict,
    )

    print(f"[validation] report={args.outdir / 'report.md'}")
    print(f"[validation] adoption={verdict['adoption']}")


def _render_preset_metrics(run_metrics: pd.DataFrame, outpath: Path) -> None:
    metrics = (
        ("mean_spread", "Mean spread"),
        ("realized_vol", "Realized vol"),
        ("abs_return_acf1", "Abs return ACF(1)"),
        ("imbalance_next_return_corr", "Imbalance -> next return"),
        ("events_per_step", "Events / step"),
        ("market_buy_share", "Market buy share"),
    )
    presets = list(dict.fromkeys(run_metrics["preset"]))
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        data = [run_metrics.loc[run_metrics["preset"] == preset, metric].dropna().to_numpy() for preset in presets]
        axis.boxplot(data, labels=presets, patch_artist=True)
        axis.set_title(label)
        axis.grid(alpha=0.25, linestyle="--")
        if metric == "market_buy_share":
            axis.axhline(0.5, color="#0f172a", linewidth=0.9, alpha=0.4)
    figure.suptitle("Preset validation metrics", fontsize=15, fontweight="bold")
    figure.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_phase_profiles(preset_summary: pd.DataFrame, outpath: Path) -> None:
    figure, (spread_ax, fill_ax) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    phase_x = list(range(len(PHASE_ORDER)))
    for _, row in preset_summary.iterrows():
        spread_values = [row[f"phase_{phase}_spread_mean"] for phase in PHASE_ORDER]
        fill_values = [row[f"phase_{phase}_fill_mean"] for phase in PHASE_ORDER]
        spread_ax.plot(phase_x, spread_values, marker="o", linewidth=1.8, label=row["preset"])
        fill_ax.plot(phase_x, fill_values, marker="o", linewidth=1.8, label=row["preset"])

    for axis, title, ylabel in (
        (spread_ax, "Session-phase spread", "Mean spread"),
        (fill_ax, "Session-phase fill activity", "Filled volume / step"),
    ):
        axis.set_xticks(phase_x)
        axis.set_xticklabels([phase.capitalize() for phase in PHASE_ORDER])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linestyle="--")
        axis.legend(frameon=False)

    figure.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_sensitivity(sensitivity_summary: pd.DataFrame, outpath: Path) -> None:
    metrics = (
        ("shock_scale", "shock_to_calm_abs_return_ratio_mean", "Shock ratio"),
        ("excitation_scale", "event_clustering_mean", "Event clustering"),
        ("meta_order_scale", "meta_directionality_mean", "Meta directionality"),
        ("fair_price_vol_scale", "realized_vol_mean", "Realized vol"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for axis, (knob_name, metric, title) in zip(axes.flat, metrics):
        frame = sensitivity_summary.loc[sensitivity_summary["knob_name"] == knob_name].sort_values("knob_scale")
        if frame.empty:
            axis.set_axis_off()
            continue
        axis.plot(frame["knob_scale"], frame[metric], marker="o", linewidth=1.8, color="#2563eb")
        axis.set_title(title)
        axis.set_xlabel(knob_name)
        axis.set_ylabel(metric.replace("_mean", ""))
        axis.grid(alpha=0.25, linestyle="--")
    figure.suptitle("Knob sensitivity summary", fontsize=15, fontweight="bold")
    figure.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_diagnostics_examples(
    *,
    presets: Sequence[str],
    seed: int,
    steps: int,
    outdir: Path,
) -> None:
    for preset in presets:
        run = run_market_validation(preset=preset, seed=seed, steps=steps)
        figure = run.market.plot_diagnostics(title=f"{preset} diagnostics", figsize=(14, 8.5))
        figure.savefig(outdir / f"diagnostics-{preset}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def _write_report(
    *,
    outpath: Path,
    presets: Sequence[str],
    seeds: Sequence[int],
    steps: int,
    run_metrics: pd.DataFrame,
    preset_summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    verdict: dict[str, object],
) -> None:
    lines = [
        "# Orderwave 검증 보고서",
        "",
        "## 실험 설정",
        f"- presets: {', '.join(presets)}",
        f"- seeds: {len(seeds)} runs per preset (`{seeds[0]}` .. `{seeds[-1]}`)",
        f"- steps per run: `{steps}`",
        f"- outputs: `get_history()`, `get_event_history()`, `get_debug_history()`, `plot_diagnostics()`",
        "",
        "## preset 요약",
        _markdown_table(
            preset_summary[
                [
                    "preset",
                    "runs",
                    "mean_spread_mean",
                    "realized_vol_mean",
                    "abs_return_acf1_mean",
                    "imbalance_next_return_corr_mean",
                    "events_per_step_mean",
                    "market_buy_share_mean",
                    "steps_per_second_mean",
                    "invariant_failures",
                ]
            ].round(4)
        ),
        "",
        "## 재현성",
        _markdown_table(reproducibility),
        "",
    ]

    if not sensitivity_summary.empty:
        lines.extend(
            [
                "## knob 민감도",
                _markdown_table(
                    sensitivity_summary[
                        [
                            "knob_name",
                            "knob_scale",
                            "realized_vol_mean",
                            "event_clustering_mean",
                            "shock_to_calm_abs_return_ratio_mean",
                            "meta_directionality_mean",
                        ]
                    ].round(4)
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 최종 판정",
            f"- synthetic market-state generator로는 `{_ko_suitability(verdict['synthetic_market_state_generator'])}`",
            f"- 주요 강점: {', '.join(verdict['major_strengths'])}",
            f"- 주요 약점: {', '.join(verdict['major_weaknesses'])}",
            f"- 실험용/연구용으로 채택 여부: `{verdict['adoption']}`",
            "",
            "## 세부 판정",
            f"- preset separation: `{_pass_fail(verdict['preset_separation_ok'])}`",
            f"- seed stability: `{_pass_fail(verdict['seed_stability_ok'])}`",
            f"- time structure: `{_pass_fail(verdict['time_structure_ok'])}`",
            f"- invariants: `{_pass_fail(verdict['invariants_ok'])}`",
            f"- reproducibility: `{_pass_fail(verdict['reproducible_ok'])}`",
            f"- sensitivity: `{_pass_fail(verdict['sensitivity_ok'])}`",
            "",
            "## 산출물",
            "- `validation-runs.csv`",
            "- `validation-summary.csv`",
            "- `validation-reproducibility.csv`",
            "- `preset-metrics.png`",
            "- `phase-profiles.png`",
        ]
    )
    if not sensitivity_summary.empty:
        lines.extend(
            [
                "- `validation-sensitivity-runs.csv`",
                "- `validation-sensitivity-summary.csv`",
                "- `sensitivity.png`",
            ]
        )
    for preset in presets:
        lines.append(f"- `diagnostics-{preset}.png`")

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = [str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _ko_suitability(value: object) -> str:
    mapping = {
        "suitable": "적합",
        "conditionally suitable": "조건부 적합",
        "not suitable": "부적합",
    }
    return mapping.get(str(value), str(value))


def _pass_fail(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"


if __name__ == "__main__":
    main()
