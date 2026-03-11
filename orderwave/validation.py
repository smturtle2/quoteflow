from __future__ import annotations

"""Compatibility facade for the orderwave validation pipeline."""

from orderwave._validation.baseline import (
    compare_validation_baseline,
    extract_validation_baseline,
    load_validation_baseline,
    write_validation_baseline,
)
from orderwave._validation.decision import evaluate_validation_results
from orderwave._validation.metrics import (
    collect_invariant_failures,
    compute_run_metrics,
    summarize_sensitivity_grid,
    summarize_validation_grid,
)
from orderwave._validation.perf import benchmark_logging_modes, measure_performance
from orderwave._validation.pipeline import run_validation_pipeline
from orderwave._validation.reproducibility import run_reproducibility_checks
from orderwave._validation.sensitivity import run_sensitivity_grid
from orderwave._validation.shared import (
    BASELINE_THROUGHPUT_FLOOR,
    BYTES_PER_LOGGED_EVENT_BUDGET,
    CORE_SENSITIVITY_KNOBS,
    DEFAULT_PRESETS,
    DEFAULT_SENSITIVITY_KNOBS,
    DEFAULT_SENSITIVITY_SCALES,
    INVARIANT_FAILURE_COLUMNS,
    PHASE_ORDER,
    SOAK_PEAK_MEMORY_BUDGET_MB,
    ValidationPipelineResult,
    ValidationRun,
)
from orderwave._validation.single_run import run_market_validation

__all__ = [
    "BASELINE_THROUGHPUT_FLOOR",
    "BYTES_PER_LOGGED_EVENT_BUDGET",
    "CORE_SENSITIVITY_KNOBS",
    "DEFAULT_PRESETS",
    "DEFAULT_SENSITIVITY_KNOBS",
    "DEFAULT_SENSITIVITY_SCALES",
    "INVARIANT_FAILURE_COLUMNS",
    "PHASE_ORDER",
    "SOAK_PEAK_MEMORY_BUDGET_MB",
    "ValidationPipelineResult",
    "ValidationRun",
    "benchmark_logging_modes",
    "collect_invariant_failures",
    "compare_validation_baseline",
    "compute_run_metrics",
    "evaluate_validation_results",
    "extract_validation_baseline",
    "load_validation_baseline",
    "measure_performance",
    "run_market_validation",
    "run_reproducibility_checks",
    "run_sensitivity_grid",
    "run_validation_pipeline",
    "summarize_sensitivity_grid",
    "summarize_validation_grid",
    "write_validation_baseline",
]
