from __future__ import annotations

from orderwave.validation import (
    compute_run_metrics,
    evaluate_validation_results,
    run_market_validation,
    run_reproducibility_checks,
    run_sensitivity_grid,
    run_validation_grid,
    summarize_sensitivity_grid,
    summarize_validation_grid,
)


def test_compute_run_metrics_smoke() -> None:
    run = run_market_validation(preset="balanced", seed=5, steps=80)
    metrics = compute_run_metrics(run, preset="balanced", seed=5, steps=80)

    assert metrics["preset"] == "balanced"
    assert metrics["seed"] == 5
    assert metrics["invariants_ok"] is True
    assert metrics["events_per_step"] > 0.0
    assert 0.0 < metrics["market_buy_share"] < 1.0
    assert metrics["steps_per_second"] > 0.0


def test_validation_grid_and_summary_smoke() -> None:
    run_metrics = run_validation_grid(
        presets=("balanced", "trend"),
        seeds=(1, 2),
        steps=60,
    )
    summary = summarize_validation_grid(run_metrics)
    reproducibility = run_reproducibility_checks(
        presets=("balanced", "trend"),
        seed=3,
        steps=30,
    )

    assert len(run_metrics) == 4
    assert set(summary["preset"]) == {"balanced", "trend"}
    assert reproducibility["all_reproducible"].all()


def test_sensitivity_summary_and_verdict_smoke() -> None:
    run_metrics = run_validation_grid(
        presets=("balanced", "trend", "volatile"),
        seeds=(1, 2),
        steps=50,
    )
    summary = summarize_validation_grid(run_metrics)
    reproducibility = run_reproducibility_checks(
        presets=("balanced", "trend", "volatile"),
        seed=4,
        steps=25,
    )
    sensitivity_runs = run_sensitivity_grid(
        preset="balanced",
        seeds=(1, 2),
        steps=40,
        knobs=("shock_scale",),
        scales=(1.0, 1.2),
    )
    sensitivity_summary = summarize_sensitivity_grid(sensitivity_runs)
    verdict = evaluate_validation_results(
        run_metrics=run_metrics,
        preset_summary=summary,
        reproducibility=reproducibility,
        sensitivity_summary=sensitivity_summary,
    )

    assert len(sensitivity_summary) == 2
    assert verdict["adoption"] in {"YES", "CONDITIONAL", "NO"}
    assert "major_strengths" in verdict
    assert "major_weaknesses" in verdict
