from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .shared import (
    DEFAULT_PRESETS,
    VALIDATION_BASELINE_METRIC_RULES,
    VALIDATION_BASELINE_SCHEMA_VERSION,
    ValidationPipelineResult,
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
        "market_identity": "aggregate_market_state_simulator",
        "validation_profile": result.profile_name,
        "liquidity_backstop_default": "on_empty",
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
        "next_focus": "market_state_fidelity",
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
    if baseline.get("market_identity") is not None and actual.get("market_identity") != baseline.get("market_identity"):
        failures.append(
            f"market_identity: expected {baseline.get('market_identity')!r}, got {actual.get('market_identity')!r}"
        )
    if baseline.get("validation_profile") is not None and actual.get("validation_profile") != baseline.get("validation_profile"):
        failures.append(
            f"validation_profile: expected {baseline.get('validation_profile')!r}, got {actual.get('validation_profile')!r}"
        )

    return {
        "matches": not failures,
        "failures": failures,
        "actual": actual,
    }
