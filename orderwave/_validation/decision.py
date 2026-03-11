from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import summarize_validation_grid
from .shared import (
    BASELINE_THROUGHPUT_FLOOR,
    BYTES_PER_LOGGED_EVENT_BUDGET,
    CORE_SENSITIVITY_KNOBS,
    EPSILON,
    SOAK_PEAK_MEMORY_BUDGET_MB,
    coefficient_of_variation,
    preset_compare,
    safe_mean,
    safe_std,
)


def evaluate_validation_results(
    *,
    run_metrics: pd.DataFrame,
    preset_summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    invariant_failures: pd.DataFrame,
    profile_name: str = "quality_regression",
) -> dict[str, Any]:
    """Produce the final validation verdict for a named validation profile."""

    baseline_metrics = run_metrics.loc[run_metrics["stage"] == "baseline"].copy()
    soak_metrics = run_metrics.loc[run_metrics["stage"] == "soak"].copy()
    baseline_summary = preset_summary.loc[preset_summary["stage"] == "baseline"].set_index("preset")
    soak_summary = preset_summary.loc[preset_summary["stage"] == "soak"].set_index("preset")

    invariants_ok = invariant_failures.empty
    reproducible_ok = bool(not reproducibility.empty and reproducibility["all_reproducible"].all())
    performance_checks: dict[str, bool] = {
        "soak_failures": bool(not soak_metrics.empty and (~soak_metrics["run_failed"]).all()),
    }
    for preset, floor in BASELINE_THROUGHPUT_FLOOR.items():
        performance_checks[f"{preset}_throughput_floor"] = bool(
            preset in baseline_summary.index and float(baseline_summary.loc[preset, "steps_per_second_mean"]) >= floor
        )
    for preset, budget in SOAK_PEAK_MEMORY_BUDGET_MB.items():
        performance_checks[f"{preset}_peak_rss_budget"] = bool(
            preset in soak_summary.index and float(soak_summary.loc[preset, "peak_memory_mb_mean"]) < budget
        )
    performance_checks["soak_bytes_per_logged_event_budget"] = bool(
        not soak_summary.empty and (soak_summary["bytes_per_logged_event_mean"] < BYTES_PER_LOGGED_EVENT_BUDGET).all()
    )
    performance_ok = all(performance_checks.values())

    classifier_accuracy = leave_one_out_centroid_accuracy(
        baseline_metrics,
        features=(
            "realized_vol",
            "mean_spread",
            "trade_sign_acf1",
            "events_per_step",
            "spread_q90",
            "one_sided_step_ratio",
            "regime_dwell_mean",
        ),
    )
    spread_q90_column = _first_available_column(baseline_summary, ("spread_q90_mean", "mean_spread_mean"))
    preset_checks = {
        "volatile_realized_vol_gt_balanced": preset_compare(baseline_summary, "volatile", "balanced", "realized_vol_mean"),
        "volatile_spread_q90_gt_balanced": preset_compare(
            baseline_summary,
            "volatile",
            "balanced",
            spread_q90_column,
            comparison="gt",
        ),
        "trend_trade_sign_acf1_gt_balanced": preset_compare(
            baseline_summary,
            "trend",
            "balanced",
            "trade_sign_acf1_mean",
        ),
        "preset_classifier_accuracy": classifier_accuracy >= 0.75,
    }
    preset_separation_ok = all(preset_checks.values())

    stylized_checks = evaluate_stylized_facts(baseline_summary)
    mandatory_stylized = (
        stylized_checks["abs_return_clustering"],
        stylized_checks["phase_structure"],
        stylized_checks["shock_response"],
        stylized_checks["impact_decay"],
    )
    stylized_ok = bool(sum(bool(value) for value in stylized_checks.values()) >= 7 and all(mandatory_stylized))

    sensitivity_checks = evaluate_sensitivity_summary(sensitivity_summary)
    core_passes = sum(int(sensitivity_checks.get(knob, False)) for knob in CORE_SENSITIVITY_KNOBS)
    total_passes = sum(int(value) for value in sensitivity_checks.values())
    sensitivity_ok = bool(core_passes == len(CORE_SENSITIVITY_KNOBS) and total_passes >= 6)

    seed_stability = evaluate_seed_stability(baseline_metrics, baseline_summary)
    seed_stability_ok = bool(seed_stability["passes"])

    hard_gates_ok = invariants_ok and reproducible_ok and performance_ok
    soft_gates_ok = preset_separation_ok and stylized_ok and sensitivity_ok and seed_stability_ok
    if profile_name == "release_smoke":
        decision = "PASS" if hard_gates_ok else "FAIL"
        suitability = "구조 안정성 통과" if hard_gates_ok else "구조 안정성 실패"
    elif not hard_gates_ok:
        decision = "NO-GO"
        suitability = "부적합"
    elif soft_gates_ok:
        decision = "GO"
        suitability = "적합"
    else:
        decision = "CONDITIONAL"
        suitability = "조건부 적합"

    strengths: list[str] = []
    weaknesses: list[str] = []
    if invariants_ok:
        strengths.append("구조적 불변식 위반이 없고 로그 정합성이 유지됨")
    else:
        weaknesses.append("구조적 불변식 또는 summary non-finite 문제가 존재함")
    if reproducible_ok:
        strengths.append("동일 seed 재실행과 gen/step 경로가 일치함")
    else:
        weaknesses.append("재현성 검사가 실패함")
    if preset_separation_ok:
        strengths.append("preset별 경로 특성이 핵심 지표 공간에서 분리됨")
    else:
        weaknesses.append("preset 분리가 약하거나 classifier 분리도가 부족함")
    if stylized_ok:
        strengths.append("변동성 군집, phase 구조, shock 반응, impact decay가 함께 관찰됨")
    else:
        weaknesses.append("stylized fact 또는 시간 구조가 충분히 드러나지 않음")
    if sensitivity_ok:
        strengths.append("노출된 knob가 대체로 의도한 방향으로 출력 분포를 제어함")
    else:
        weaknesses.append("민감도 반응이 약하거나 일부 knob 방향성이 뒤집힘")
    if seed_stability_ok:
        strengths.append("seed 변화에도 preset-level 결론이 유지됨")
    else:
        weaknesses.append("seed 민감도가 높아 정성적 결론이 흔들림")
    if performance_ok:
        strengths.append("반복 실험용 처리량과 장기 soak 안정성이 확보됨")
    else:
        weaknesses.append("성능 floor 또는 장기 soak 메모리 예산을 넘김")

    if profile_name == "release_smoke":
        immediate_scope = "release regression smoke check"
    else:
        immediate_scope = (
            "preset 비교, multi-seed 실험, 시장상태 경로 연구, 전략 프로토타이핑"
            if decision != "NO-GO"
            else "추가 수정 전에는 연구용 채택 범위를 권장하지 않음"
        )
    required_fix = (
        "없음"
        if decision in {"GO", "PASS"}
        else ", ".join(weaknesses[:2]) if weaknesses else "추가 검증 필요"
    )

    if profile_name == "release_smoke":
        conclusion_market_state = (
            "이 검증은 시장상태 시뮬레이터의 구조 안정성 smoke regression만 평가한다."
        )
    else:
        conclusion_market_state = (
            f"이 엔진은 aggregate order-book 기반 시장상태 시뮬레이터로 볼 때 {suitability}."
        )

    return {
        "profile_name": profile_name,
        "decision": decision,
        "suitability": suitability,
        "invariants_ok": invariants_ok,
        "reproducibility_ok": reproducible_ok,
        "performance_ok": performance_ok,
        "preset_separation_ok": preset_separation_ok,
        "stylized_facts_ok": stylized_ok,
        "sensitivity_ok": sensitivity_ok,
        "seed_stability_ok": seed_stability_ok,
        "performance_checks": performance_checks,
        "preset_checks": {**preset_checks, "classifier_accuracy": float(classifier_accuracy)},
        "stylized_checks": stylized_checks,
        "sensitivity_checks": sensitivity_checks,
        "seed_stability": seed_stability,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "immediate_scope": immediate_scope,
        "required_fix": required_fix,
        "conclusion_market_state": conclusion_market_state,
        "conclusion_scope": f"현재 가장 신뢰할 수 있는 사용처는 {immediate_scope}.",
        "conclusion_required_fix": f"채택 전 반드시 보완할 항목은 {required_fix}.",
    }


def evaluate_stylized_facts(baseline_summary: pd.DataFrame) -> dict[str, bool]:
    if baseline_summary.empty:
        return {
            "abs_return_clustering": False,
            "spread_variation": False,
            "imbalance_signal": False,
            "phase_structure": False,
            "event_clustering": False,
            "shock_response": False,
            "impact_decay": False,
            "depletion_recovery": False,
            "regime_persistence": False,
            "meta_directionality": False,
        }

    abs_return_acf1 = _summary_series(baseline_summary, "abs_return_acf1_mean")
    spread_unique = _summary_series(baseline_summary, "spread_unique_count_mean")
    imbalance_signal = _summary_series(baseline_summary, "imbalance_next_mid_return_corr_mean")
    phase_spread_range = _summary_series(baseline_summary, "phase_spread_range_mean")
    phase_fill_range = _summary_series(baseline_summary, "phase_fill_range_mean")
    buy_event_acf1 = _summary_series(baseline_summary, "buy_event_count_acf1_mean")
    cancel_event_acf1 = _summary_series(baseline_summary, "cancel_event_count_acf1_mean")
    shock_ratio = _summary_series(baseline_summary, "shock_to_calm_ratio_mean")
    shock_decay = _summary_series(baseline_summary, "shock_decay_ratio_mean")
    meta_decay = _summary_series(baseline_summary, "meta_decay_ratio_mean")
    regime_dwell = _summary_series(baseline_summary, "regime_dwell_mean")
    meta_active = _summary_series(baseline_summary, "meta_active_directional_ratio_mean")
    meta_inactive = _summary_series(baseline_summary, "meta_inactive_directional_ratio_mean")

    return {
        "abs_return_clustering": bool((abs_return_acf1 > 0.0).all()),
        "spread_variation": bool((spread_unique >= 3.0).all()),
        "imbalance_signal": bool((imbalance_signal > 0.0).all()),
        "phase_structure": bool(
            (phase_spread_range > 0.001).any()
            or (phase_fill_range > 1.0).any()
        ),
        "event_clustering": bool(
            (
                (buy_event_acf1 > 0.0)
                & (cancel_event_acf1 > 0.0)
            ).sum()
            >= 2
        ),
        "shock_response": bool((shock_ratio > 1.0).sum() >= 2),
        "impact_decay": bool(
            ((shock_decay > 0.0) & (shock_decay < 1.0)).sum()
            >= 2
            and ((meta_decay > 0.0) & (meta_decay < 1.0)).sum()
            >= 2
        ),
        "depletion_recovery": bool((_summary_series(baseline_summary, "depletion_recovery_half_life_mean") > 0.0).all()),
        "regime_persistence": bool((regime_dwell >= 2.0).sum() >= 2),
        "meta_directionality": bool(
            (meta_active > meta_inactive).sum()
            >= 2
        ),
    }


def evaluate_sensitivity_summary(sensitivity_summary: pd.DataFrame) -> dict[str, bool]:
    if sensitivity_summary.empty:
        return {}
    checks: dict[str, bool] = {}
    for knob_name, frame in sensitivity_summary.groupby("knob_name", sort=False):
        checks[knob_name] = bool(frame["direction_ok"].iloc[0])
    return checks


def evaluate_seed_stability(baseline_metrics: pd.DataFrame, baseline_summary: pd.DataFrame) -> dict[str, Any]:
    if baseline_metrics.empty or baseline_summary.empty:
        return {"passes": False, "details": {}, "qualitative_conclusions_hold": False}

    core_metrics = (
        "mean_spread",
        "realized_vol",
        "events_per_step",
        "imbalance_next_mid_return_corr",
        "abs_return_acf1",
    )
    details: dict[str, Any] = {}
    passes = True
    for preset, frame in baseline_metrics.groupby("preset", sort=False):
        row: dict[str, float] = {}
        high_cv_count = 0
        for metric in core_metrics:
            if metric not in frame.columns:
                row[f"{metric}_cv"] = 0.0
                continue
            mean_value = safe_mean(frame[metric])
            std_value = safe_std(frame[metric])
            cv_value = coefficient_of_variation(mean_value, std_value)
            row[f"{metric}_cv"] = cv_value
            if cv_value > 0.5:
                high_cv_count += 1
        row["high_cv_count"] = float(high_cv_count)
        details[preset] = row
        if high_cv_count > 2:
            passes = False

    trimmed = remove_worst_seed_per_preset(baseline_metrics, core_metrics)
    trimmed_summary = summarize_validation_grid(trimmed)
    full_index = baseline_summary
    trimmed_index = trimmed_summary.set_index("preset")
    qualitative_conclusions_hold = bool(
        preset_compare(full_index, "volatile", "balanced", "realized_vol_mean")
        == preset_compare(trimmed_index, "volatile", "balanced", "realized_vol_mean")
        and preset_compare(full_index, "volatile", "balanced", "mean_spread_mean", comparison="ge")
        == preset_compare(trimmed_index, "volatile", "balanced", "mean_spread_mean", comparison="ge")
        and preset_compare(full_index, "trend", "balanced", "trade_sign_acf1_mean")
        == preset_compare(trimmed_index, "trend", "balanced", "trade_sign_acf1_mean")
    )
    return {
        "passes": bool(passes and qualitative_conclusions_hold),
        "details": details,
        "qualitative_conclusions_hold": qualitative_conclusions_hold,
    }


def _first_available_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    return candidates[0]


def _summary_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(float)
    return pd.Series(0.0, index=frame.index, dtype=float)


def remove_worst_seed_per_preset(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    keep_rows = []
    for _, group in frame.groupby("preset", sort=False):
        if len(group) <= 1:
            keep_rows.append(group.copy())
            continue
        zscores = np.zeros(len(group), dtype=float)
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=float)
            std = float(np.std(values))
            if std <= EPSILON:
                continue
            zscores += np.abs((values - float(np.mean(values))) / std)
        drop_position = int(np.argmax(zscores))
        keep_rows.append(group.drop(group.index[drop_position]))
    return pd.concat(keep_rows, ignore_index=True, sort=False)


def leave_one_out_centroid_accuracy(frame: pd.DataFrame, *, features: Sequence[str]) -> float:
    if frame.empty or frame["preset"].nunique() < 2:
        return 0.0
    usable_features = [feature for feature in features if feature in frame.columns]
    if not usable_features:
        return 0.0
    data = frame[usable_features].to_numpy(dtype=float)
    labels = frame["preset"].to_numpy()
    correct = 0
    total = 0
    for index in range(len(frame)):
        train_mask = np.ones(len(frame), dtype=bool)
        train_mask[index] = False
        train = data[train_mask]
        train_labels = labels[train_mask]
        if len(np.unique(train_labels)) < 2:
            continue
        mean = np.mean(train, axis=0)
        std = np.std(train, axis=0)
        std[std <= EPSILON] = 1.0
        train_scaled = (train - mean) / std
        test_scaled = (data[index] - mean) / std
        centroids = {
            label: np.mean(train_scaled[train_labels == label], axis=0)
            for label in np.unique(train_labels)
        }
        predicted = min(centroids.items(), key=lambda item: float(np.linalg.norm(test_scaled - item[1])))[0]
        correct += int(predicted == labels[index])
        total += 1
    return float(correct / max(total, 1))
