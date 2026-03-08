from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .shared import DEFAULT_SENSITIVITY_SCALES, markdown_bullets, markdown_table, pass_fail


def write_validation_summary(
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
        f"- invariants: `{pass_fail(acceptance['invariants_ok'])}`",
        f"- reproducibility: `{pass_fail(acceptance['reproducibility_ok'])}`",
        f"- performance: `{pass_fail(acceptance['performance_ok'])}`",
        "",
        "### 성능 체크",
        markdown_bullets(acceptance["performance_checks"]),
        "",
        "## 3. baseline preset summary",
        markdown_table(
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
        markdown_table(reproducibility) if not reproducibility.empty else "_no data_",
        "",
        "## 5. sensitivity summary",
        markdown_table(
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
        markdown_table(
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
        f"- preset separation: `{pass_fail(acceptance['preset_separation_ok'])}`",
        f"- stylized facts / time structure: `{pass_fail(acceptance['stylized_facts_ok'])}`",
        f"- sensitivity: `{pass_fail(acceptance['sensitivity_ok'])}`",
        f"- seed stability: `{pass_fail(acceptance['seed_stability_ok'])}`",
        "",
        "### preset separation details",
        markdown_bullets(acceptance["preset_checks"]),
        "",
        "### stylized facts",
        markdown_bullets(acceptance["stylized_checks"]),
        "",
        "### sensitivity direction checks",
        markdown_bullets(acceptance["sensitivity_checks"]),
        "",
        "## 8. invariant failures",
        f"- failure rows: `{len(invariant_failures)}`",
        markdown_table(invariant_failures.head(20)) if not invariant_failures.empty else "_none_",
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
            "- 평가 관점: synthetic market-state generator",
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


def write_acceptance_decision(*, outpath: Path, acceptance: Mapping[str, Any]) -> None:
    lines = [
        "# Acceptance Decision",
        "",
        f"- final verdict: `{acceptance['decision']}`",
        f"- invariants: `{pass_fail(acceptance['invariants_ok'])}`",
        f"- reproducibility: `{pass_fail(acceptance['reproducibility_ok'])}`",
        f"- preset separation: `{pass_fail(acceptance['preset_separation_ok'])}`",
        f"- stylized facts: `{pass_fail(acceptance['stylized_facts_ok'])}`",
        f"- sensitivity: `{pass_fail(acceptance['sensitivity_ok'])}`",
        f"- seed stability: `{pass_fail(acceptance['seed_stability_ok'])}`",
        f"- performance: `{pass_fail(acceptance['performance_ok'])}`",
        "",
        acceptance["conclusion_market_state"],
        acceptance["conclusion_scope"],
        acceptance["conclusion_required_fix"],
        "",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")
