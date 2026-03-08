from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from orderwave.validation import (
    benchmark_logging_modes,
    compare_validation_baseline,
    collect_invariant_failures,
    compute_run_metrics,
    evaluate_validation_results,
    extract_validation_baseline,
    load_validation_baseline,
    measure_performance,
    run_market_validation,
    run_reproducibility_checks,
    run_validation_pipeline,
    summarize_sensitivity_grid,
    write_validation_baseline,
)


def test_compute_run_metrics_respects_warmup() -> None:
    run = run_market_validation(preset="balanced", seed=5, steps=40, warmup_fraction=0.25)
    metrics = compute_run_metrics(run, stage="baseline", preset="balanced", seed=5)

    assert metrics["warmup_cutoff_step"] == 10
    assert metrics["analysis_steps"] == 31
    assert metrics["run_failed"] is False
    assert metrics["mean_spread"] > 0.0
    assert metrics["events_per_step"] > 0.0
    assert metrics["bytes_per_logged_event"] > 0.0
    assert metrics["knob_scale"] is None
    assert metrics["repeat_idx"] is None


def test_collect_invariant_failures_empty_for_smoke_run() -> None:
    run = run_market_validation(preset="balanced", seed=7, steps=60)
    failures = collect_invariant_failures(run, stage="baseline", preset="balanced", seed=7)

    assert list(failures.columns) == [
        "preset",
        "seed",
        "stage",
        "step",
        "event_idx",
        "invariant_name",
        "details",
    ]
    assert failures.empty


def test_reproducibility_hash_and_step_equivalence() -> None:
    reproducibility = run_reproducibility_checks(
        presets=("balanced", "trend"),
        seed=3,
        steps=30,
    )

    assert len(reproducibility) == 2
    assert reproducibility["history_hash_equal"].all()
    assert reproducibility["event_hash_equal"].all()
    assert reproducibility["debug_hash_equal"].all()
    assert reproducibility["metrics_hash_equal"].all()
    assert reproducibility["gen_vs_step_history_equal"].all()
    assert reproducibility["gen_vs_step_event_equal"].all()
    assert reproducibility["gen_vs_step_debug_equal"].all()


def test_sensitivity_summary_direction_scoring() -> None:
    metrics = pd.DataFrame(
        [
            {
                "knob_name": "shock_scale",
                "knob_scale": 0.5,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.3,
                "shock_to_calm_ratio": 1.1,
            },
            {
                "knob_name": "shock_scale",
                "knob_scale": 1.0,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.3,
                "shock_to_calm_ratio": 1.3,
            },
            {
                "knob_name": "shock_scale",
                "knob_scale": 1.5,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.3,
                "shock_to_calm_ratio": 1.5,
            },
            {
                "knob_name": "shock_scale",
                "knob_scale": 2.0,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.3,
                "shock_to_calm_ratio": 1.8,
            },
            {
                "knob_name": "meta_order_scale",
                "knob_scale": 0.5,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.6,
                "shock_to_calm_ratio": 1.1,
            },
            {
                "knob_name": "meta_order_scale",
                "knob_scale": 1.0,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.5,
                "shock_to_calm_ratio": 1.1,
            },
            {
                "knob_name": "meta_order_scale",
                "knob_scale": 1.5,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.4,
                "shock_to_calm_ratio": 1.1,
            },
            {
                "knob_name": "meta_order_scale",
                "knob_scale": 2.0,
                "limit_share": 0.2,
                "events_per_step_market": 0.5,
                "cancel_share": 0.4,
                "realized_vol": 0.8,
                "regime_switch_rate": 0.1,
                "phase_spread_range": 0.1,
                "phase_fill_range": 0.2,
                "buy_event_count_acf1": 0.1,
                "cancel_event_count_acf1": 0.2,
                "meta_active_directional_ratio": 0.3,
                "shock_to_calm_ratio": 1.1,
            },
        ]
    )

    summary = summarize_sensitivity_grid(metrics)

    assert bool(summary.loc[summary["knob_name"] == "shock_scale", "direction_ok"].iloc[0]) is True
    assert bool(summary.loc[summary["knob_name"] == "meta_order_scale", "direction_ok"].iloc[0]) is False


def test_benchmark_logging_modes_shows_history_only_reduction() -> None:
    frame = benchmark_logging_modes(preset="balanced", seed=11, steps=120)

    assert set(frame["logging_mode"]) == {"full", "history_only"}
    assert bool(frame.loc[frame["logging_mode"] == "full", "run_failed"].iloc[0]) is False
    assert bool(frame.loc[frame["logging_mode"] == "history_only", "run_failed"].iloc[0]) is False
    assert frame.loc[frame["logging_mode"] == "history_only", "logged_rows"].iloc[0] == 0


def test_measure_performance_returns_summary_and_seed_metrics() -> None:
    result = measure_performance(preset="balanced", seeds=(1, 2), steps=80)

    assert set(result) == {"seed_metrics", "summary", "logging_compare"}
    assert len(result["seed_metrics"]) == 2
    assert len(result["summary"]) == 1
    assert set(result["logging_compare"]["logging_mode"]) == {"full", "history_only"}
    assert "floor_pass" in result["summary"].columns
    assert result["summary"]["throughput_floor"].iloc[0] > 0.0


def test_validation_baseline_round_trip_and_self_compare(tmp_path: Path) -> None:
    result = run_validation_pipeline(
        outdir=tmp_path / "validation",
        baseline_seeds=1,
        baseline_steps=30,
        sensitivity_seeds=1,
        sensitivity_steps=20,
        long_run_seeds=1,
        long_run_steps=40,
        jobs=1,
    )

    baseline_path = tmp_path / "validation_baseline.json"
    write_validation_baseline(baseline_path, result)
    baseline = load_validation_baseline(baseline_path)
    comparison = compare_validation_baseline(result, baseline)

    assert baseline["schema_version"] == 1
    assert baseline["liquidity_backstop_default"] == "always"
    assert comparison["matches"] is True
    assert comparison["failures"] == []


def test_validation_baseline_detects_drift(tmp_path: Path) -> None:
    result = run_validation_pipeline(
        outdir=tmp_path / "validation",
        baseline_seeds=1,
        baseline_steps=30,
        sensitivity_seeds=1,
        sensitivity_steps=20,
        long_run_seeds=1,
        long_run_steps=40,
        jobs=1,
    )
    baseline = extract_validation_baseline(result)
    baseline["acceptance"]["decision"] = "GO"

    comparison = compare_validation_baseline(result, baseline)

    assert comparison["matches"] is False
    assert any("acceptance.decision" in failure for failure in comparison["failures"])


def test_readme_mentions_liquidity_backstop_default() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert 'Default `liquidity_backstop="always"` keeps the synthetic market two-sided and observable by default.' in readme


def test_repo_validation_baseline_file_has_expected_shape() -> None:
    baseline = load_validation_baseline(Path("tests/golden/validation_baseline.json"))

    assert baseline["schema_version"] == 1
    assert baseline["liquidity_backstop_default"] == "always"
    assert baseline["acceptance"]["decision"] == "GO"
    assert baseline["next_focus"] == "finer_event_feedback"


def test_release_validation_baseline_file_has_expected_shape() -> None:
    baseline = load_validation_baseline(Path("tests/golden/validation_release_baseline.json"))

    assert baseline["schema_version"] == 1
    assert baseline["liquidity_backstop_default"] == "always"
    assert baseline["next_focus"] == "finer_event_feedback"


def test_run_validation_pipeline_writes_final_artifacts(tmp_path: Path) -> None:
    result = run_validation_pipeline(
        outdir=tmp_path,
        baseline_seeds=1,
        baseline_steps=30,
        sensitivity_seeds=1,
        sensitivity_steps=20,
        long_run_seeds=1,
        long_run_steps=40,
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
    assert {path.name for path in result.diagnostics_paths.values()} == {
        "diagnostics_balanced_1.png",
        "diagnostics_trend_1.png",
        "diagnostics_volatile_1.png",
    }


def test_decision_engine_mappings_use_new_performance_gate() -> None:
    run_metrics = pd.DataFrame(
        [
            {
                "stage": "baseline",
                "preset": "balanced",
                "seed": 1,
                "realized_vol": 0.0100,
                "mean_spread": 0.0200,
                "trade_sign_acf1": 0.050,
                "events_per_step": 30.0,
                "abs_return_acf1": 0.10,
                "imbalance_next_mid_return_corr": 0.10,
                "spread_unique_count": 4.0,
                "phase_spread_range": 0.01,
                "phase_fill_range": 10.0,
                "buy_event_count_acf1": 0.2,
                "cancel_event_count_acf1": 0.2,
                "shock_to_calm_ratio": 1.2,
                "meta_active_directional_ratio": 0.3,
                "meta_inactive_directional_ratio": 0.1,
                "steps_per_second": 500.0,
                "peak_memory_mb": 400.0,
                "bytes_per_logged_event": 120.0,
                "run_failed": False,
                "memory_growth_without_recovery": True,
            },
            {
                "stage": "baseline",
                "preset": "trend",
                "seed": 1,
                "realized_vol": 0.0120,
                "mean_spread": 0.0220,
                "trade_sign_acf1": 0.090,
                "events_per_step": 45.0,
                "abs_return_acf1": 0.12,
                "imbalance_next_mid_return_corr": 0.11,
                "spread_unique_count": 6.0,
                "phase_spread_range": 0.02,
                "phase_fill_range": 20.0,
                "buy_event_count_acf1": 0.3,
                "cancel_event_count_acf1": 0.3,
                "shock_to_calm_ratio": 1.3,
                "meta_active_directional_ratio": 0.5,
                "meta_inactive_directional_ratio": 0.1,
                "steps_per_second": 350.0,
                "peak_memory_mb": 700.0,
                "bytes_per_logged_event": 180.0,
                "run_failed": False,
                "memory_growth_without_recovery": True,
            },
            {
                "stage": "baseline",
                "preset": "volatile",
                "seed": 1,
                "realized_vol": 0.0200,
                "mean_spread": 0.0300,
                "trade_sign_acf1": 0.070,
                "events_per_step": 60.0,
                "abs_return_acf1": 0.09,
                "imbalance_next_mid_return_corr": 0.14,
                "spread_unique_count": 5.0,
                "phase_spread_range": 0.015,
                "phase_fill_range": 15.0,
                "buy_event_count_acf1": 0.2,
                "cancel_event_count_acf1": 0.2,
                "shock_to_calm_ratio": 1.4,
                "meta_active_directional_ratio": 0.25,
                "meta_inactive_directional_ratio": 0.1,
                "steps_per_second": 300.0,
                "peak_memory_mb": 900.0,
                "bytes_per_logged_event": 190.0,
                "run_failed": False,
                "memory_growth_without_recovery": True,
            },
            {
                "stage": "soak",
                "preset": "balanced",
                "seed": 1,
                "realized_vol": 0.0100,
                "mean_spread": 0.0200,
                "trade_sign_acf1": 0.050,
                "events_per_step": 30.0,
                "abs_return_acf1": 0.10,
                "imbalance_next_mid_return_corr": 0.10,
                "spread_unique_count": 4.0,
                "phase_spread_range": 0.01,
                "phase_fill_range": 10.0,
                "buy_event_count_acf1": 0.2,
                "cancel_event_count_acf1": 0.2,
                "shock_to_calm_ratio": 1.2,
                "meta_active_directional_ratio": 0.3,
                "meta_inactive_directional_ratio": 0.1,
                "steps_per_second": 480.0,
                "peak_memory_mb": 1800.0,
                "bytes_per_logged_event": 150.0,
                "run_failed": False,
                "memory_growth_without_recovery": True,
            },
            {
                "stage": "soak",
                "preset": "trend",
                "seed": 1,
                "realized_vol": 0.0120,
                "mean_spread": 0.0220,
                "trade_sign_acf1": 0.090,
                "events_per_step": 45.0,
                "abs_return_acf1": 0.12,
                "imbalance_next_mid_return_corr": 0.11,
                "spread_unique_count": 6.0,
                "phase_spread_range": 0.02,
                "phase_fill_range": 20.0,
                "buy_event_count_acf1": 0.3,
                "cancel_event_count_acf1": 0.3,
                "shock_to_calm_ratio": 1.3,
                "meta_active_directional_ratio": 0.5,
                "meta_inactive_directional_ratio": 0.1,
                "steps_per_second": 320.0,
                "peak_memory_mb": 2500.0,
                "bytes_per_logged_event": 200.0,
                "run_failed": False,
                "memory_growth_without_recovery": True,
            },
            {
                "stage": "soak",
                "preset": "volatile",
                "seed": 1,
                "realized_vol": 0.0200,
                "mean_spread": 0.0300,
                "trade_sign_acf1": 0.070,
                "events_per_step": 60.0,
                "abs_return_acf1": 0.09,
                "imbalance_next_mid_return_corr": 0.14,
                "spread_unique_count": 5.0,
                "phase_spread_range": 0.015,
                "phase_fill_range": 15.0,
                "buy_event_count_acf1": 0.2,
                "cancel_event_count_acf1": 0.2,
                "shock_to_calm_ratio": 1.4,
                "meta_active_directional_ratio": 0.25,
                "meta_inactive_directional_ratio": 0.1,
                "steps_per_second": 280.0,
                "peak_memory_mb": 3000.0,
                "bytes_per_logged_event": 210.0,
                "run_failed": False,
                "memory_growth_without_recovery": True,
            },
        ]
    )
    preset_summary = pd.DataFrame(
        [
            {
                "stage": "baseline",
                "preset": "balanced",
                "runs": 1,
                "run_failures": 0,
                "memory_growth_failures": 1,
                "mean_spread_mean": 0.0200,
                "realized_vol_mean": 0.0100,
                "trade_sign_acf1_mean": 0.050,
                "events_per_step_mean": 30.0,
                "abs_return_acf1_mean": 0.10,
                "imbalance_next_mid_return_corr_mean": 0.10,
                "spread_unique_count_mean": 4.0,
                "phase_spread_range_mean": 0.01,
                "phase_fill_range_mean": 10.0,
                "buy_event_count_acf1_mean": 0.2,
                "cancel_event_count_acf1_mean": 0.2,
                "shock_to_calm_ratio_mean": 1.2,
                "meta_active_directional_ratio_mean": 0.3,
                "meta_inactive_directional_ratio_mean": 0.1,
                "steps_per_second_mean": 500.0,
                "peak_memory_mb_mean": 400.0,
                "bytes_per_logged_event_mean": 120.0,
            },
            {
                "stage": "baseline",
                "preset": "trend",
                "runs": 1,
                "run_failures": 0,
                "memory_growth_failures": 1,
                "mean_spread_mean": 0.0220,
                "realized_vol_mean": 0.0120,
                "trade_sign_acf1_mean": 0.090,
                "events_per_step_mean": 45.0,
                "abs_return_acf1_mean": 0.12,
                "imbalance_next_mid_return_corr_mean": 0.11,
                "spread_unique_count_mean": 6.0,
                "phase_spread_range_mean": 0.02,
                "phase_fill_range_mean": 20.0,
                "buy_event_count_acf1_mean": 0.3,
                "cancel_event_count_acf1_mean": 0.3,
                "shock_to_calm_ratio_mean": 1.3,
                "meta_active_directional_ratio_mean": 0.5,
                "meta_inactive_directional_ratio_mean": 0.1,
                "steps_per_second_mean": 350.0,
                "peak_memory_mb_mean": 700.0,
                "bytes_per_logged_event_mean": 180.0,
            },
            {
                "stage": "baseline",
                "preset": "volatile",
                "runs": 1,
                "run_failures": 0,
                "memory_growth_failures": 1,
                "mean_spread_mean": 0.0300,
                "realized_vol_mean": 0.0200,
                "trade_sign_acf1_mean": 0.070,
                "events_per_step_mean": 60.0,
                "abs_return_acf1_mean": 0.09,
                "imbalance_next_mid_return_corr_mean": 0.14,
                "spread_unique_count_mean": 5.0,
                "phase_spread_range_mean": 0.015,
                "phase_fill_range_mean": 15.0,
                "buy_event_count_acf1_mean": 0.2,
                "cancel_event_count_acf1_mean": 0.2,
                "shock_to_calm_ratio_mean": 1.4,
                "meta_active_directional_ratio_mean": 0.25,
                "meta_inactive_directional_ratio_mean": 0.1,
                "steps_per_second_mean": 300.0,
                "peak_memory_mb_mean": 900.0,
                "bytes_per_logged_event_mean": 190.0,
            },
            {
                "stage": "soak",
                "preset": "balanced",
                "runs": 1,
                "run_failures": 0,
                "memory_growth_failures": 1,
                "mean_spread_mean": 0.0200,
                "realized_vol_mean": 0.0100,
                "trade_sign_acf1_mean": 0.050,
                "events_per_step_mean": 30.0,
                "abs_return_acf1_mean": 0.10,
                "imbalance_next_mid_return_corr_mean": 0.10,
                "spread_unique_count_mean": 4.0,
                "phase_spread_range_mean": 0.01,
                "phase_fill_range_mean": 10.0,
                "buy_event_count_acf1_mean": 0.2,
                "cancel_event_count_acf1_mean": 0.2,
                "shock_to_calm_ratio_mean": 1.2,
                "meta_active_directional_ratio_mean": 0.3,
                "meta_inactive_directional_ratio_mean": 0.1,
                "steps_per_second_mean": 480.0,
                "peak_memory_mb_mean": 1800.0,
                "bytes_per_logged_event_mean": 150.0,
            },
            {
                "stage": "soak",
                "preset": "trend",
                "runs": 1,
                "run_failures": 0,
                "memory_growth_failures": 1,
                "mean_spread_mean": 0.0220,
                "realized_vol_mean": 0.0120,
                "trade_sign_acf1_mean": 0.090,
                "events_per_step_mean": 45.0,
                "abs_return_acf1_mean": 0.12,
                "imbalance_next_mid_return_corr_mean": 0.11,
                "spread_unique_count_mean": 6.0,
                "phase_spread_range_mean": 0.02,
                "phase_fill_range_mean": 20.0,
                "buy_event_count_acf1_mean": 0.3,
                "cancel_event_count_acf1_mean": 0.3,
                "shock_to_calm_ratio_mean": 1.3,
                "meta_active_directional_ratio_mean": 0.5,
                "meta_inactive_directional_ratio_mean": 0.1,
                "steps_per_second_mean": 320.0,
                "peak_memory_mb_mean": 2500.0,
                "bytes_per_logged_event_mean": 200.0,
            },
            {
                "stage": "soak",
                "preset": "volatile",
                "runs": 1,
                "run_failures": 0,
                "memory_growth_failures": 1,
                "mean_spread_mean": 0.0300,
                "realized_vol_mean": 0.0200,
                "trade_sign_acf1_mean": 0.070,
                "events_per_step_mean": 60.0,
                "abs_return_acf1_mean": 0.09,
                "imbalance_next_mid_return_corr_mean": 0.14,
                "spread_unique_count_mean": 5.0,
                "phase_spread_range_mean": 0.015,
                "phase_fill_range_mean": 15.0,
                "buy_event_count_acf1_mean": 0.2,
                "cancel_event_count_acf1_mean": 0.2,
                "shock_to_calm_ratio_mean": 1.4,
                "meta_active_directional_ratio_mean": 0.25,
                "meta_inactive_directional_ratio_mean": 0.1,
                "steps_per_second_mean": 280.0,
                "peak_memory_mb_mean": 3000.0,
                "bytes_per_logged_event_mean": 210.0,
            },
        ]
    )
    reproducibility = pd.DataFrame(
        [
            {
                "preset": "balanced",
                "all_reproducible": True,
            },
            {
                "preset": "trend",
                "all_reproducible": True,
            },
            {
                "preset": "volatile",
                "all_reproducible": True,
            },
        ]
    )
    sensitivity_summary = pd.DataFrame(
        [
            {"knob_name": "limit_rate_scale", "direction_ok": True},
            {"knob_name": "market_rate_scale", "direction_ok": True},
            {"knob_name": "cancel_rate_scale", "direction_ok": True},
            {"knob_name": "fair_price_vol_scale", "direction_ok": True},
            {"knob_name": "regime_transition_scale", "direction_ok": True},
            {"knob_name": "seasonality_scale", "direction_ok": True},
            {"knob_name": "excitation_scale", "direction_ok": True},
            {"knob_name": "meta_order_scale", "direction_ok": True},
            {"knob_name": "shock_scale", "direction_ok": True},
        ]
    )

    acceptance = evaluate_validation_results(
        run_metrics=run_metrics,
        preset_summary=preset_summary,
        reproducibility=reproducibility,
        sensitivity_summary=sensitivity_summary,
        invariant_failures=pd.DataFrame(columns=["preset", "seed", "stage", "step", "event_idx", "invariant_name", "details"]),
    )

    assert acceptance["performance_ok"] is True
    assert acceptance["decision"] in {"GO", "CONDITIONAL"}
