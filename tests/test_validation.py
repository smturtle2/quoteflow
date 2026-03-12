from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from orderwave.validation import (
    compare_validation_baseline,
    compute_run_metrics,
    evaluate_validation_results,
    load_validation_baseline,
    run_market_validation,
    run_validation_pipeline,
    summarize_sensitivity_grid,
    write_validation_baseline,
)


def test_compute_run_metrics_smoke() -> None:
    run = run_market_validation(preset="balanced", seed=5, steps=20, warmup_fraction=0.20)
    metrics = compute_run_metrics(run, stage="baseline", preset="balanced", seed=5)

    assert metrics["warmup_cutoff_step"] == 4
    assert metrics["analysis_steps"] > 0
    assert metrics["run_failed"] is False
    assert metrics["mean_spread"] > 0.0
    assert metrics["events_per_step"] > 0.0


def test_sensitivity_summary_direction_scoring() -> None:
    metrics = pd.DataFrame(
        [
            {"knob_name": "shock_scale", "knob_scale": 0.5, "limit_share": 0.2, "events_per_step_market": 0.5, "cancel_share": 0.4, "realized_vol": 0.8, "regime_switch_rate": 0.1, "phase_spread_range": 0.1, "phase_fill_range": 0.2, "buy_event_count_acf1": 0.1, "cancel_event_count_acf1": 0.2, "meta_active_directional_ratio": 0.3, "meta_active_impact_ratio": 0.9, "meta_impact_ratio": 0.9, "shock_to_calm_ratio": 1.1, "shock_impact_ratio": 1.1, "return_tail_ratio": 1.3, "one_sided_ratio": 0.02, "one_sided_step_ratio": 0.02},
            {"knob_name": "shock_scale", "knob_scale": 1.0, "limit_share": 0.2, "events_per_step_market": 0.5, "cancel_share": 0.4, "realized_vol": 0.8, "regime_switch_rate": 0.1, "phase_spread_range": 0.1, "phase_fill_range": 0.2, "buy_event_count_acf1": 0.1, "cancel_event_count_acf1": 0.2, "meta_active_directional_ratio": 0.3, "meta_active_impact_ratio": 0.9, "meta_impact_ratio": 0.9, "shock_to_calm_ratio": 1.3, "shock_impact_ratio": 1.3, "return_tail_ratio": 1.3, "one_sided_ratio": 0.02, "one_sided_step_ratio": 0.02},
            {"knob_name": "shock_scale", "knob_scale": 1.5, "limit_share": 0.2, "events_per_step_market": 0.5, "cancel_share": 0.4, "realized_vol": 0.8, "regime_switch_rate": 0.1, "phase_spread_range": 0.1, "phase_fill_range": 0.2, "buy_event_count_acf1": 0.1, "cancel_event_count_acf1": 0.2, "meta_active_directional_ratio": 0.3, "meta_active_impact_ratio": 0.9, "meta_impact_ratio": 0.9, "shock_to_calm_ratio": 1.5, "shock_impact_ratio": 1.5, "return_tail_ratio": 1.3, "one_sided_ratio": 0.02, "one_sided_step_ratio": 0.02},
            {"knob_name": "shock_scale", "knob_scale": 2.0, "limit_share": 0.2, "events_per_step_market": 0.5, "cancel_share": 0.4, "realized_vol": 0.8, "regime_switch_rate": 0.1, "phase_spread_range": 0.1, "phase_fill_range": 0.2, "buy_event_count_acf1": 0.1, "cancel_event_count_acf1": 0.2, "meta_active_directional_ratio": 0.3, "meta_active_impact_ratio": 0.9, "meta_impact_ratio": 0.9, "shock_to_calm_ratio": 1.8, "shock_impact_ratio": 1.8, "return_tail_ratio": 1.3, "one_sided_ratio": 0.02, "one_sided_step_ratio": 0.02},
        ]
    )

    summary = summarize_sensitivity_grid(metrics)

    assert bool(summary.loc[summary["knob_name"] == "shock_scale", "direction_ok"].iloc[0]) is True


def test_validation_pipeline_smoke_and_baseline_round_trip(tmp_path: Path) -> None:
    result = run_validation_pipeline(
        outdir=tmp_path,
        baseline_seeds=1,
        baseline_steps=12,
        sensitivity_seeds=1,
        sensitivity_steps=8,
        long_run_seeds=1,
        long_run_steps=16,
        jobs=1,
    )

    expected_files = {
        "validation_summary.md",
        "run_metrics.csv",
        "preset_summary.csv",
        "sensitivity_summary.csv",
        "invariant_failures.csv",
        "acceptance_decision.md",
    }
    assert expected_files <= {path.name for path in tmp_path.iterdir() if path.is_file()}

    baseline_path = tmp_path / "validation_baseline.json"
    write_validation_baseline(baseline_path, result)
    baseline = load_validation_baseline(baseline_path)
    comparison = compare_validation_baseline(result, baseline)

    assert baseline["schema_version"] == 2
    assert baseline["validation_profile"] == "quality_regression"
    assert baseline["market_identity"] == "aggregate_market_state_simulator"
    assert baseline["liquidity_backstop_default"] == "on_empty"
    assert comparison["matches"] is True


def test_readme_mentions_liquidity_backstop_default() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert 'liquidity_backstop="on_empty"' in readme


def test_validation_baseline_files_have_expected_shape() -> None:
    repo_baseline = load_validation_baseline(Path("tests/golden/validation_baseline.json"))
    release_baseline = load_validation_baseline(Path("tests/golden/validation_release_baseline.json"))

    assert repo_baseline["schema_version"] == 2
    assert repo_baseline["market_identity"] == "aggregate_market_state_simulator"
    assert repo_baseline["validation_profile"] == "quality_regression"
    assert repo_baseline["liquidity_backstop_default"] == "on_empty"
    assert repo_baseline["next_focus"] == "market_state_fidelity"
    assert release_baseline["schema_version"] == 2
    assert release_baseline["market_identity"] == "aggregate_market_state_simulator"
    assert release_baseline["validation_profile"] == "release_smoke"
    assert release_baseline["liquidity_backstop_default"] == "on_empty"
    assert release_baseline["next_focus"] == "market_state_fidelity"


def test_decision_engine_performance_gate_smoke() -> None:
    def metric_row(stage: str, preset: str, *, realized_vol: float, mean_spread: float, trade_sign_acf1: float, events_per_step: float, spread_q90: float, one_sided_step_ratio: float, meta_impact_ratio: float, shock_impact_ratio: float, regime_dwell_mean: float, steps_per_second: float, peak_memory_mb: float, bytes_per_logged_event: float) -> dict[str, object]:
        return {
            "stage": stage,
            "preset": preset,
            "seed": 1,
            "realized_vol": realized_vol,
            "mean_spread": mean_spread,
            "trade_sign_acf1": trade_sign_acf1,
            "events_per_step": events_per_step,
            "abs_return_acf1": 0.10,
            "imbalance_next_mid_return_corr": 0.10,
            "spread_unique_count": 4.0,
            "phase_spread_range": 0.01,
            "phase_fill_range": 10.0,
            "buy_event_count_acf1": 0.2,
            "cancel_event_count_acf1": 0.2,
            "shock_to_calm_ratio": shock_impact_ratio,
            "shock_impact_ratio": shock_impact_ratio,
            "shock_decay_ratio": 0.6,
            "meta_active_directional_ratio": 0.3,
            "meta_inactive_directional_ratio": 0.1,
            "meta_active_impact_ratio": meta_impact_ratio,
            "meta_impact_ratio": meta_impact_ratio,
            "meta_decay_ratio": 0.6,
            "steps_per_second": steps_per_second,
            "peak_memory_mb": peak_memory_mb,
            "bytes_per_logged_event": bytes_per_logged_event,
            "run_failed": False,
            "memory_growth_without_recovery": False,
            "spread_q90": spread_q90,
            "spread_q99": spread_q90 + 0.01,
            "return_tail_ratio": 1.6 if preset == "volatile" else 1.2,
            "one_sided_step_ratio": one_sided_step_ratio,
            "one_sided_ratio": one_sided_step_ratio,
            "depletion_recovery_half_life": 2.0,
            "regime_dwell_mean": regime_dwell_mean,
        }

    run_metrics = pd.DataFrame(
        [
            metric_row("baseline", "balanced", realized_vol=0.01, mean_spread=0.02, trade_sign_acf1=0.05, events_per_step=30.0, spread_q90=0.03, one_sided_step_ratio=0.02, meta_impact_ratio=1.1, shock_impact_ratio=1.2, regime_dwell_mean=2.5, steps_per_second=400.0, peak_memory_mb=400.0, bytes_per_logged_event=120.0),
            metric_row("baseline", "trend", realized_vol=0.012, mean_spread=0.022, trade_sign_acf1=0.09, events_per_step=45.0, spread_q90=0.035, one_sided_step_ratio=0.03, meta_impact_ratio=1.3, shock_impact_ratio=1.3, regime_dwell_mean=3.0, steps_per_second=280.0, peak_memory_mb=700.0, bytes_per_logged_event=180.0),
            metric_row("baseline", "volatile", realized_vol=0.02, mean_spread=0.03, trade_sign_acf1=0.07, events_per_step=60.0, spread_q90=0.05, one_sided_step_ratio=0.08, meta_impact_ratio=1.2, shock_impact_ratio=1.4, regime_dwell_mean=3.2, steps_per_second=240.0, peak_memory_mb=900.0, bytes_per_logged_event=190.0),
            metric_row("soak", "balanced", realized_vol=0.01, mean_spread=0.02, trade_sign_acf1=0.05, events_per_step=30.0, spread_q90=0.03, one_sided_step_ratio=0.02, meta_impact_ratio=1.1, shock_impact_ratio=1.2, regime_dwell_mean=2.5, steps_per_second=380.0, peak_memory_mb=1800.0, bytes_per_logged_event=150.0),
            metric_row("soak", "trend", realized_vol=0.012, mean_spread=0.022, trade_sign_acf1=0.09, events_per_step=45.0, spread_q90=0.035, one_sided_step_ratio=0.03, meta_impact_ratio=1.3, shock_impact_ratio=1.3, regime_dwell_mean=3.0, steps_per_second=260.0, peak_memory_mb=2500.0, bytes_per_logged_event=200.0),
            metric_row("soak", "volatile", realized_vol=0.02, mean_spread=0.03, trade_sign_acf1=0.07, events_per_step=60.0, spread_q90=0.05, one_sided_step_ratio=0.08, meta_impact_ratio=1.2, shock_impact_ratio=1.4, regime_dwell_mean=3.2, steps_per_second=220.0, peak_memory_mb=3000.0, bytes_per_logged_event=210.0),
        ]
    )
    preset_summary = pd.DataFrame(
        [
            {"stage": "baseline", "preset": "balanced", "mean_spread_mean": 0.02, "realized_vol_mean": 0.01, "trade_sign_acf1_mean": 0.05, "events_per_step_mean": 30.0, "abs_return_acf1_mean": 0.10, "imbalance_next_mid_return_corr_mean": 0.10, "spread_unique_count_mean": 4.0, "phase_spread_range_mean": 0.01, "phase_fill_range_mean": 10.0, "buy_event_count_acf1_mean": 0.2, "cancel_event_count_acf1_mean": 0.2, "shock_to_calm_ratio_mean": 1.2, "shock_decay_ratio_mean": 0.6, "meta_active_directional_ratio_mean": 0.3, "meta_inactive_directional_ratio_mean": 0.1, "meta_active_impact_ratio_mean": 1.1, "meta_impact_ratio_mean": 1.1, "meta_decay_ratio_mean": 0.6, "spread_q90_mean": 0.03, "return_tail_ratio_mean": 1.2, "one_sided_step_ratio_mean": 0.02, "regime_dwell_mean": 2.5, "depletion_recovery_half_life_mean": 2.0, "steps_per_second_mean": 400.0, "peak_memory_mb_mean": 400.0, "bytes_per_logged_event_mean": 120.0, "runs": 1, "run_failures": 0, "memory_growth_failures": 0},
            {"stage": "baseline", "preset": "trend", "mean_spread_mean": 0.022, "realized_vol_mean": 0.012, "trade_sign_acf1_mean": 0.09, "events_per_step_mean": 45.0, "abs_return_acf1_mean": 0.12, "imbalance_next_mid_return_corr_mean": 0.11, "spread_unique_count_mean": 6.0, "phase_spread_range_mean": 0.02, "phase_fill_range_mean": 20.0, "buy_event_count_acf1_mean": 0.3, "cancel_event_count_acf1_mean": 0.3, "shock_to_calm_ratio_mean": 1.3, "shock_decay_ratio_mean": 0.6, "meta_active_directional_ratio_mean": 0.5, "meta_inactive_directional_ratio_mean": 0.1, "meta_active_impact_ratio_mean": 1.3, "meta_impact_ratio_mean": 1.3, "meta_decay_ratio_mean": 0.6, "spread_q90_mean": 0.035, "return_tail_ratio_mean": 1.25, "one_sided_step_ratio_mean": 0.03, "regime_dwell_mean": 3.0, "depletion_recovery_half_life_mean": 2.0, "steps_per_second_mean": 280.0, "peak_memory_mb_mean": 700.0, "bytes_per_logged_event_mean": 180.0, "runs": 1, "run_failures": 0, "memory_growth_failures": 0},
            {"stage": "baseline", "preset": "volatile", "mean_spread_mean": 0.03, "realized_vol_mean": 0.02, "trade_sign_acf1_mean": 0.07, "events_per_step_mean": 60.0, "abs_return_acf1_mean": 0.09, "imbalance_next_mid_return_corr_mean": 0.14, "spread_unique_count_mean": 5.0, "phase_spread_range_mean": 0.015, "phase_fill_range_mean": 15.0, "buy_event_count_acf1_mean": 0.2, "cancel_event_count_acf1_mean": 0.2, "shock_to_calm_ratio_mean": 1.4, "shock_decay_ratio_mean": 0.6, "meta_active_directional_ratio_mean": 0.25, "meta_inactive_directional_ratio_mean": 0.1, "meta_active_impact_ratio_mean": 1.2, "meta_impact_ratio_mean": 1.2, "meta_decay_ratio_mean": 0.6, "spread_q90_mean": 0.05, "return_tail_ratio_mean": 1.6, "one_sided_step_ratio_mean": 0.08, "regime_dwell_mean": 3.2, "depletion_recovery_half_life_mean": 2.0, "steps_per_second_mean": 240.0, "peak_memory_mb_mean": 900.0, "bytes_per_logged_event_mean": 190.0, "runs": 1, "run_failures": 0, "memory_growth_failures": 0},
            {"stage": "soak", "preset": "balanced", "mean_spread_mean": 0.02, "realized_vol_mean": 0.01, "trade_sign_acf1_mean": 0.05, "events_per_step_mean": 30.0, "abs_return_acf1_mean": 0.10, "imbalance_next_mid_return_corr_mean": 0.10, "spread_unique_count_mean": 4.0, "phase_spread_range_mean": 0.01, "phase_fill_range_mean": 10.0, "buy_event_count_acf1_mean": 0.2, "cancel_event_count_acf1_mean": 0.2, "shock_to_calm_ratio_mean": 1.2, "shock_decay_ratio_mean": 0.6, "meta_active_directional_ratio_mean": 0.3, "meta_inactive_directional_ratio_mean": 0.1, "meta_active_impact_ratio_mean": 1.1, "meta_impact_ratio_mean": 1.1, "meta_decay_ratio_mean": 0.6, "spread_q90_mean": 0.03, "return_tail_ratio_mean": 1.2, "one_sided_step_ratio_mean": 0.02, "regime_dwell_mean": 2.5, "depletion_recovery_half_life_mean": 2.0, "steps_per_second_mean": 380.0, "peak_memory_mb_mean": 1800.0, "bytes_per_logged_event_mean": 150.0, "runs": 1, "run_failures": 0, "memory_growth_failures": 0},
            {"stage": "soak", "preset": "trend", "mean_spread_mean": 0.022, "realized_vol_mean": 0.012, "trade_sign_acf1_mean": 0.09, "events_per_step_mean": 45.0, "abs_return_acf1_mean": 0.12, "imbalance_next_mid_return_corr_mean": 0.11, "spread_unique_count_mean": 6.0, "phase_spread_range_mean": 0.02, "phase_fill_range_mean": 20.0, "buy_event_count_acf1_mean": 0.3, "cancel_event_count_acf1_mean": 0.3, "shock_to_calm_ratio_mean": 1.3, "shock_decay_ratio_mean": 0.6, "meta_active_directional_ratio_mean": 0.5, "meta_inactive_directional_ratio_mean": 0.1, "meta_active_impact_ratio_mean": 1.3, "meta_impact_ratio_mean": 1.3, "meta_decay_ratio_mean": 0.6, "spread_q90_mean": 0.035, "return_tail_ratio_mean": 1.25, "one_sided_step_ratio_mean": 0.03, "regime_dwell_mean": 3.0, "depletion_recovery_half_life_mean": 2.0, "steps_per_second_mean": 260.0, "peak_memory_mb_mean": 2500.0, "bytes_per_logged_event_mean": 200.0, "runs": 1, "run_failures": 0, "memory_growth_failures": 0},
            {"stage": "soak", "preset": "volatile", "mean_spread_mean": 0.03, "realized_vol_mean": 0.02, "trade_sign_acf1_mean": 0.07, "events_per_step_mean": 60.0, "abs_return_acf1_mean": 0.09, "imbalance_next_mid_return_corr_mean": 0.14, "spread_unique_count_mean": 5.0, "phase_spread_range_mean": 0.015, "phase_fill_range_mean": 15.0, "buy_event_count_acf1_mean": 0.2, "cancel_event_count_acf1_mean": 0.2, "shock_to_calm_ratio_mean": 1.4, "shock_decay_ratio_mean": 0.6, "meta_active_directional_ratio_mean": 0.25, "meta_inactive_directional_ratio_mean": 0.1, "meta_active_impact_ratio_mean": 1.2, "meta_impact_ratio_mean": 1.2, "meta_decay_ratio_mean": 0.6, "spread_q90_mean": 0.05, "return_tail_ratio_mean": 1.6, "one_sided_step_ratio_mean": 0.08, "regime_dwell_mean": 3.2, "depletion_recovery_half_life_mean": 2.0, "steps_per_second_mean": 220.0, "peak_memory_mb_mean": 3000.0, "bytes_per_logged_event_mean": 210.0, "runs": 1, "run_failures": 0, "memory_growth_failures": 0},
        ]
    )
    reproducibility = pd.DataFrame([{"preset": "balanced", "all_reproducible": True}, {"preset": "trend", "all_reproducible": True}, {"preset": "volatile", "all_reproducible": True}])
    sensitivity_summary = pd.DataFrame([{"knob_name": name, "direction_ok": True} for name in ("limit_rate_scale", "market_rate_scale", "cancel_rate_scale", "fair_price_vol_scale", "regime_transition_scale", "seasonality_scale", "excitation_scale", "meta_order_scale", "shock_scale")])

    acceptance = evaluate_validation_results(
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        reproducibility=reproducibility,
        sensitivity_summary=sensitivity_summary,
        invariant_failures=pd.DataFrame(columns=["preset", "seed", "stage", "step", "event_idx", "invariant_name", "details"]),
    )

    assert acceptance["performance_ok"] is True
    assert acceptance["decision"] in {"GO", "CONDITIONAL"}

    release_run_metrics = run_metrics.copy()
    release_run_metrics.loc[
        (release_run_metrics["stage"] == "baseline") & (release_run_metrics["preset"] == "balanced"),
        "steps_per_second",
    ] = 220.0
    release_run_metrics.loc[
        (release_run_metrics["stage"] == "baseline") & (release_run_metrics["preset"] == "trend"),
        "steps_per_second",
    ] = 190.0
    release_run_metrics.loc[
        (release_run_metrics["stage"] == "baseline") & (release_run_metrics["preset"] == "volatile"),
        "steps_per_second",
    ] = 170.0
    release_preset_summary = preset_summary.copy()
    release_preset_summary.loc[
        (release_preset_summary["stage"] == "baseline") & (release_preset_summary["preset"] == "balanced"),
        "steps_per_second_mean",
    ] = 220.0
    release_preset_summary.loc[
        (release_preset_summary["stage"] == "baseline") & (release_preset_summary["preset"] == "trend"),
        "steps_per_second_mean",
    ] = 190.0
    release_preset_summary.loc[
        (release_preset_summary["stage"] == "baseline") & (release_preset_summary["preset"] == "volatile"),
        "steps_per_second_mean",
    ] = 170.0

    release_acceptance = evaluate_validation_results(
        run_metrics=release_run_metrics,
        preset_summary=release_preset_summary,
        reproducibility=reproducibility,
        sensitivity_summary=sensitivity_summary,
        invariant_failures=pd.DataFrame(columns=["preset", "seed", "stage", "step", "event_idx", "invariant_name", "details"]),
        profile_name="release_smoke",
    )

    assert release_acceptance["performance_checks"]["balanced_throughput_floor"] is True
    assert release_acceptance["performance_checks"]["trend_throughput_floor"] is True
    assert release_acceptance["performance_checks"]["volatile_throughput_floor"] is True
    assert release_acceptance["performance_ok"] is True
    assert release_acceptance["decision"] == "PASS"
